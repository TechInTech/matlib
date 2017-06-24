# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import numpy as np
import warnings
import copy
import inspect

from ..utils import _to_snake_case, get_uid, _collect_previous_mask, \
    batch_get_value, batch_set_value
from . import initializers, constraints, regularizers, activations


def _bias_add(out, bias):
    return out + bias


def _object_list_uid(object_list):
    object_list = _to_list(object_list)
    return ', '.join([str(abs(id(x))) for x in object_list])


def _is_all_none(iterable_or_element):
    if not isinstance(iterable_or_element, (list, tuple)):
        iterable = [iterable_or_element]
    else:
        iterable = iterable_or_element
    for element in iterable:
        if element is not None:
            return False
    return True


def _to_list(x):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.

    # Returns
        A list.
    """
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _collect_input_shape(input_tensors):
    if '_shape' in input_tensors:
        shapes = input_tensors['_shape']
        if len(shapes) == 1:
            return shapes[0]
        return shapes
    else:
        raise ValueError('inputs has no shape')


class InputSpec(object):

    def __init__(self, dtype=None,
                 shape=None,
                 ndim=None,
                 max_ndim=None,
                 min_ndim=None,
                 axes=None):
        self.dtype = dtype
        self.shape = shape
        if shape is not None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}


class Node(object):
    """A `Node` describes the connectivity between two layers.

    Each time a layer is connected to some new input,
    a node is added to `layer.inbound_nodes`.
    Each time the output of a layer is used by another layer,
    a node is added to `layer.outbound_nodes`.

    # Arguments
        outbound_layer: the layer that takes
            `input_tensors` and turns them into `output_tensors`
            (the node gets created when the `call`
            method of the layer was called).
        inbound_layers: a list of layers, the same length as `input_tensors`,
            the layers from where `input_tensors` originate.
        node_indices: a list of integers, the same length as `inbound_layers`.
            `node_indices[i]` is the origin node of `input_tensors[i]`
            (necessary since each inbound layer might have several nodes,
            e.g. if the layer is being shared with a different data stream).
        tensor_indices: a list of integers,
            the same length as `inbound_layers`.
            `tensor_indices[i]` is the index of `input_tensors[i]` within the
            output of the inbound layer
            (necessary since each inbound layer might
            have multiple tensor outputs, with each one being
            independently manipulable).
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        input_masks: list of input masks (a mask can be a tensor, or None).
        output_masks: list of output masks (a mask can be a tensor, or None).
        input_shapes: list of input shape tuples.
        output_shapes: list of output shape tuples.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.

    `node_indices` and `tensor_indices` are basically fine-grained coordinates
    describing the origin of the `input_tensors`, verifying the following:

    `input_tensors[i] == inbound_layers[i].inbound_nodes[
        node_indices[i]].output_tensors[tensor_indices[i]]`

    A node from layer A to layer B is added to:
        A.outbound_nodes
        B.inbound_nodes
    """

    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_masks, output_masks,
                 input_shapes, output_shapes,
                 arguments=None):
        # Layer instance (NOT a list).
        # this is the layer that takes a list of input tensors
        # and turns them into a list of output tensors.
        # the current node will be added to
        # the inbound_nodes of outbound_layer.
        self.outbound_layer = outbound_layer

        # The following 3 properties describe where
        # the input tensors come from: which layers,
        # and for each layer, which node and which
        # tensor output of each node.

        # List of layer instances.
        self.inbound_layers = inbound_layers
        # List of integers, 1:1 mapping with inbound_layers.
        self.node_indices = node_indices
        # List of integers, 1:1 mapping with inbound_layers.
        self.tensor_indices = tensor_indices

        # Following 2 properties:
        # tensor inputs and outputs of outbound_layer.

        # List of tensors. 1:1 mapping with inbound_layers.
        self.input_tensors = input_tensors
        # List of tensors, created by outbound_layer.call().
        self.output_tensors = output_tensors

        # Following 2 properties: input and output masks.
        # List of tensors, 1:1 mapping with input_tensor.
        self.input_masks = input_masks
        # List of tensors, created by outbound_layer.compute_mask().
        self.output_masks = output_masks

        # Following 2 properties: input and output shapes.

        # List of shape tuples, shapes of input_tensors.
        self.input_shapes = input_shapes
        # List of shape tuples, shapes of output_tensors.
        self.output_shapes = output_shapes

        # Optional keyword arguments to layer's `call`.
        self.arguments = arguments

        # Add nodes to all layers involved.
        for layer in inbound_layers:
            if layer is not None:
                layer.outbound_nodes.append(self)
        outbound_layer.inbound_nodes.append(self)

    def get_config(self):
        inbound_names = []
        for layer in self.inbound_layers:
            if layer:
                inbound_names.append(layer.name)
            else:
                inbound_names.append(None)
        return {'outbound_layer': self.outbound_layer.name if self.outbound_layer else None,
                'inbound_layers': inbound_names,
                'node_indices': self.node_indices,
                'tensor_indices': self.tensor_indices}


class Layer(object):

    def __init__(self, **kwargs):

        self.input_spec = None
        self.supports_masking = False

        # These properties will be set upon call of self.build()
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._constraints = {}  # dict {tensor: constraint instance}
        self._losses = []
        self._updates = []
        self._per_input_losses = {}
        self._per_input_updates = {}
        self._built = False

        # These lists will be filled via successive calls
        # to self._add_inbound_node().
        self.inbound_nodes = []
        self.outbound_nodes = []

        # These properties should be set by the user via keyword arguments.
        # note that 'dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {'input_shape',
                          'batch_input_shape',
                          'batch_size',
                          'dtype',
                          'name',
                          'trainable',
                          'weights',
                          'input_dtype',  # legacy
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + '_' + str(get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
            # In this case we will later create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                if 'batch_size' in kwargs:
                    batch_size = kwargs['batch_size']
                else:
                    batch_size = None
                batch_input_shape = (batch_size,) + \
                    tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape

            # Set dtype.
            dtype = kwargs.get('dtype')
            if dtype is None:
                dtype = kwargs.get('input_dtype')
            if dtype is None:
                dtype = 'float32'
            self.dtype = dtype

        if 'weights' in kwargs:
            self._initial_weights = kwargs['weights']
        else:
            self._initial_weights = None

    @property
    def losses(self):
        return self._losses

    @property
    def updates(self):
        return self._updates

    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, value):
        self._built = value

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        self._constraints = constraints

    @property
    def trainable_weights(self):
        trainable = getattr(self, 'trainable', True)
        if trainable:
            return self._trainable_weights
        else:
            return []

    @trainable_weights.setter
    def trainable_weights(self, weights):
        self._trainable_weights = weights

    @property
    def non_trainable_weights(self):
        trainable = getattr(self, 'trainable', True)
        if not trainable:
            return self._trainable_weights + self._non_trainable_weights
        else:
            return self._non_trainable_weights

    @non_trainable_weights.setter
    def non_trainable_weights(self, weights):
        self._non_trainable_weights = weights

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.

        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        # Returns
            A tensor or list/tuple of tensors.
        """
        return inputs

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, np.ndarray):
            inputs = {'data': (inputs), '_shape': inputs.shape}
        elif isinstance(inputs, (tuple, list)):
            is_ndarray_list = True
            inputs_shape = []
            for i in inputs:
                if not isinstance(i, np.ndarray):
                    is_nparray_list = False
                    break
                else:
                    inputs_shape.append(i.shape)
            if is_ndarray_list:
                inputs = {'data': inputs, '_shape': inputs_shape}
            else:
                raise ValueError(
                    'inputs is list but contains element which is not a ndarray')
        else:
            raise ValueError('invalid inputs')

        input_shape = _collect_input_shape(inputs)

        if not self.built:
            self.assert_input_compatibility(inputs['data'])
            self.build(input_shape)
            if self._initial_weights is not None:
                self.set_weights(self._initial_weights)
        self.assert_input_compatibility(inputs['data'])

        previous_mask = _collect_previous_mask(inputs)
        user_kwargs = copy.copy(kwargs)

        if not _is_all_none(previous_mask):
            if 'mask' in inspect.getargspec(self.call).args:
                if 'mask' not in kwargs:
                    kwargs['mask'] = previous_mask

        if all([s is not None for s in input_shape]):
            output_shape = self.compute_output_shape(input_shape)
        else:
            if isinstance(input_shape, list):
                output_shape = [None for _ in input_shape]
            else:
                output_shape = None

        # Actually call the layer, collecting output(s), mask(s), and shape(s).
        output = self.call(inputs['data'], **kwargs)
        output_mask = self.compute_mask(inputs, previous_mask)

        self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                               input_masks=previous_mask, output_masks=output_mask,
                               input_shapes=input_shape, output_shapes=output_shape,
                               arguments=user_kwargs)
        if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
            regularization_losses = [
                self.activity_regularizer(x) for x in _to_list(output)]
            self.add_loss(regularization_losses, _to_list(inputs))

        return output

    def _add_inbound_node(self, input_tensors, output_tensors,
                          input_masks, output_masks,
                          input_shapes, output_shapes, arguments=None):

        input_tensors = _to_list(input_tensors)
        output_tensors = _to_list(output_tensors)
        input_masks = _to_list(input_masks)
        output_masks = _to_list(output_masks)
        input_shapes = _to_list(input_shapes)
        output_shapes = _to_list(output_shapes)

        # Collect input tensor(s) coordinates.
        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for x in input_tensors:
            if hasattr(x, '_history'):
                inbound_layer, node_index, tensor_index = x._history
                inbound_layers.append(inbound_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers.append(None)
                node_indices.append(None)
                tensor_indices.append(None)

        # Create node, add it to inbound nodes.
        Node(
            self,
            inbound_layers=inbound_layers,
            node_indices=node_indices,
            tensor_indices=tensor_indices,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_masks=input_masks,
            output_masks=output_masks,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            arguments=arguments
        )

        # Update tensor history, _shape and _uses_learning_phase.
        # for i in range(len(output_tensors)):
        #     output_tensors[i]._shape = output_shapes[i]
        #     uses_lp = any([getattr(x, '_uses_learning_phase', False)
        #                    for x in input_tensors])
        #     uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
        #     output_tensors[i]._uses_learning_phase = getattr(
        #         output_tensors[i], '_uses_learning_phase', False) or uses_lp
        # output_tensors[i]._history = (self, len(self.inbound_nodes) - 1, i)

    def add_weight(self,
                   name,
                   shape,
                   dtype='float32',
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        initializer = initializers.get(initializer)
        weight = initializer(shape, dtype=dtype)
        if regularizer is not None:
            self.add_loss(regularizer(weight))
        if constraint is not None:
            self.constraints['weight'] = constraint
        # if trainable:
        #     self._trainable_weights.append(weight)
        # else:
        #     self._non_trainable_weights.append(weight)
        return weight

    def add_loss(self, losses, inputs=None):
        if losses is None or losses == []:
            return
        # Update self.losses
        losses = _to_list(losses)
        if hasattr(self, '_losses'):
            self._losses += losses
        # Update self._per_input_updates
        if inputs == []:
            inputs = None
        if inputs is not None:
            inputs_hash = _object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_losses:
            self._per_input_losses[inputs_hash] = []
        self._per_input_losses[inputs_hash] += losses

    def assert_input_compatibility(self, inputs):
        if not self.input_spec:
            return
        input_spec = _to_list(self.input_spec)
        inputs = _to_list(inputs)
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                             'but it received ' + str(len(inputs)) +
                             ' input tensors. Input received: ' +
                             str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec is None:
                continue

            # Check ndim.
            if spec.ndim is not None:
                if x.ndim != spec.ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(x.ndim))
            if spec.max_ndim is not None:
                ndim = x.ndim
                if ndim is not None and ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(x.ndim))
            if spec.min_ndim is not None:
                ndim = x.ndim
                if ndim is not None and ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(x.ndim))
            # Check dtype.
            if spec.dtype is not None:
                if x.dtype != spec.dtype:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected dtype=' +
                                     str(spec.dtype) + ', found dtype=' +
                                     str(x.dtype))
            # Check specific shape axes.
            if spec.axes:
                x_shape = x.shape
                if x_shape is not None:
                    for axis, value in spec.axes.items():
                        if value is not None and x_shape[int(axis)] not in {value, None}:
                            raise ValueError('Input ' + str(input_index) +
                                             ' is incompatible with layer ' +
                                             self.name + ': expected axis ' +
                                             str(axis) +
                                             ' of input shape to have '
                                             'value ' + str(value) +
                                             ' but got shape ' + str(x_shape))
            # Check shape.
            if spec.shape is not None:
                x_shape = x.shape
                if x_shape != spec.shape:
                    raise ValueError(
                        'Input ' + str(input_index) +
                        ' is incompatible with layer ' +
                        self.name + ': expected shape=' +
                        str(spec.shape) + ', found shape=' +
                        str(x_shape))

    def build(self, input_shape):

        self.built = True

    def compute_mask(self, inputs, mask):
        # check if support mask
        if not self.supports_masking:
            if mask is not None:
                if isinstance(mask, list):
                    if any(m is not None for m in mask):
                        raise TypeError('Layer ' + self.name +
                                        ' does not support masking, '
                                        'but was passed an input_mask: ' +
                                        str(mask))
                else:
                    raise TypeError('Layer ' + self.name +
                                    ' does not support masking, '
                                    'but was passed an input_mask: ' +
                                    str(mask))
            return None
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        # Retrieves an attribute (e.g. input_tensors) from a node.
        if not self.inbound_nodes:
            raise RuntimeError('The layer has never been called '
                               'and thus has no defined ' + attr_name + '.')
        if not len(self.inbound_nodes) > node_index:
            raise ValueError('Asked to get ' + attr_name +
                             ' at node ' + str(node_index) +
                             ', but the layer has only ' +
                             str(len(self.inbound_nodes)) + ' inbound nodes.')
        values = getattr(self.inbound_nodes[node_index], attr)
        if len(values) == 1:
            return values[0]
        else:
            return values

    def get_input_shape_at(self, node_index):
        # Retrieves the input shape(s) of a layer at a given node.
        return self._get_node_attribute_at_index(node_index,
                                                 'input_shapes',
                                                 'input shape')

    def get_output_shape_at(self, node_index):
        # Retrieves the output shape(s) of a layer at a given node.
        return self._get_node_attribute_at_index(node_index,
                                                 'output_shapes',
                                                 'output shape')

    def get_input_at(self, node_index):
        # Retrieves the input tensor(s) of a layer at a given node.
        return self._get_node_attribute_at_index(node_index,
                                                 'input_tensors',
                                                 'input')

    def get_output_at(self, node_index):
        # Retrieves the output tensor(s) of a layer at a given node.
        return self._get_node_attribute_at_index(node_index,
                                                 'output_tensors',
                                                 'output')

    def get_input_mask_at(self, node_index):
        # Retrieves the input mask tensor(s) of a layer at a given node.
        return self._get_node_attribute_at_index(node_index,
                                                 'input_masks',
                                                 'input mask')

    def get_output_mask_at(self, node_index):
        # Retrieves the output mask tensor(s) of a layer at a given node.
        return self._get_node_attribute_at_index(node_index,
                                                 'output_masks',
                                                 'output mask')

    def _has_only_one_inbound_nodes(self):
        if not self.inbound_nodes:
            raise AttributeError('Layer ' + self.name +
                                 ' has no inbound nodes.')
        if len(self.inbound_nodes) > 1:
            raise AttributeError('Layer ' + self.name +
                                 ' has multiple inbound nodes, '
                                 'hence the notion of "layer output" '
                                 'is ill-defined. '
                                 'Use `get_output_at(node_index)` instead.')

    @property
    def input(self):
        # Retrieves the input tensor(s) of a layer.

        return self._get_node_attribute_at_index(0, 'input_tensors',
                                                 'input')

    @property
    def output(self):
        # Retrieves the output tensor(s) of a layer.
        # Only applicable if the layer has exactly one inbound node
        self._has_only_one_inbound_nodes()
        return self._get_node_attribute_at_index(0, 'output_tensors',
                                                 'output')

    @property
    def input_mask(self):
        # Retrieves the input mask tensor(s) of a layer.
        self._has_only_one_inbound_nodes()
        return self._get_node_attribute_at_index(0, 'input_masks',
                                                 'input mask')

    @property
    def output_mask(self):
        # Retrieves the output mask tensor(s) of a layer.
        # Only applicable if the layer has exactly one inbound node
        self._has_only_one_inbound_nodes()
        return self._get_node_attribute_at_index(0, 'output_masks',
                                                 'output mask')

    @property
    def input_shape(self):
        # """Retrieves the input shape tuple(s) of a layer
        if not self.inbound_nodes:
            raise AttributeError('The layer has never been called '
                                 'and thus has no defined input shape.')
        all_input_shapes = set([str(node.input_shapes)
                                for node in self.inbound_nodes])
        if len(all_input_shapes) == 1:
            input_shapes = self.inbound_nodes[0].input_shapes
            if len(input_shapes) == 1:
                return input_shapes[0]
            else:
                return input_shapes
        else:
            raise AttributeError('The layer "' + str(self.name) +
                                 ' has multiple inbound nodes, '
                                 'with different input shapes. Hence '
                                 'the notion of "input shape" is '
                                 'ill-defined for the layer. '
                                 'Use `get_input_shape_at(node_index)` '
                                 'instead.')

    @property
    def output_shape(self):
        # """Retrieves the output shape tuple(s) of a layer.
        if not self.inbound_nodes:
            raise AttributeError('The layer has never been called '
                                 'and thus has no defined output shape.')
        all_output_shapes = set([str(node.output_shapes)
                                 for node in self.inbound_nodes])
        if len(all_output_shapes) == 1:
            output_shapes = self.inbound_nodes[0].output_shapes
            if len(output_shapes) == 1:
                return output_shapes[0]
            else:
                return output_shapes
        else:
            raise AttributeError('The layer "' + str(self.name) +
                                 ' has multiple inbound nodes, '
                                 'with different output shapes. Hence '
                                 'the notion of "output shape" is '
                                 'ill-defined for the layer. '
                                 'Use `get_output_shape_at(node_index)` '
                                 'instead.')

    @property
    def weights(self):
        return self.trainable_weights + self.non_trainable_weights

    def get_weights(self):
        return batch_get_value(self.weights)

    def set_weights(self, weights):
        params = self.weights
        if len(params) != len(weights):
            raise ValueError('You called `set_weights(weights)` on layer "' +
                             self.name +
                             '" with a  weight list of length ' +
                             str(len(weights)) +
                             ', but the layer was expecting ' +
                             str(len(params)) +
                             ' weights. Provided weights: ' +
                             str(weights)[:50] + '...')
        if not params:
            return
        weight_value_tuples = []
        param_values = batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Layer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        batch_set_value(weight_value_tuples)

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable}
        if hasattr(self, 'batch_input_shape'):
            config['batch_input_shape'] = self.batch_input_shape
        if hasattr(self, 'dtype'):
            config['dtype'] = self.dtype
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def count_params(self):
        return sum([p.size for p in self.weights])


class Dense(Layer):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel', shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        out = np.dot(inputs, self.kernel)
        if self.use_bias:
            out = _bias_add(out, self.bias)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ActivityRegularization(Layer):

    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super(ActivityRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2
        self.activity_regularizer = regularizers.L1L2(l1=l1, l2=l2)

    def get_config(self):
        config = {'l1': self.l1,
                  'l2': self.l2}
        base_config = super(ActivityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):

    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, inputs):
        return inputs.reshape(inputs.shape[0], np.prod(inputs.shape[1:]))


class Dropout(Layer):
    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = np.clip(rate, 0., 1.)
        self.supports_masking = True

    def call(self, inputs):
        rnd = np.random.uniform(size=(inputs.shape[0]))
        out = inputs[rnd > self.rate]
        out /= (1.0-self.rate)

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Activation(Layer):

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))