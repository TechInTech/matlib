import numpy as np
import six
import inspect
import re
# keras object storage

_GLOBAL_CUSTOM_OBJECTS = {}
_UID_PREFIXES = {}

def _collect_previous_mask(inputs):
    masks = []
    for elem in inputs:
        if hasattr(elem, '_history'):
            inbound_layer, node_index, tensor_index = elem._history
            node = inbound_layer.inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        else:
            masks.append(None)
    if len(masks) == 1:
        return masks[0]
    return masks

def get_value(x):
    if not hasattr(x, 'get_value'):
        raise TypeError('get_value() can only be called on a variable. '
                        'If you have an expression instead, use eval().')
    return x.get_value()


def batch_get_value(xs):
    """Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    """
    return [get_value(x) for x in xs]

def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def batch_set_value(tuples):
    for x, value in tuples:
        x.set_value(np.asarray(value, dtype=x.dtype))
        
def _to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure

def get_uid(prefix=''):
    """Provides a unique UID given a string prefix.

    # Arguments
        prefix: string.

    # Returns
        An integer.

    # Example
    ```
        >>> keras.backend.get_uid('dense')
        >>> 1
        >>> keras.backend.get_uid('dense')
        >>> 2
    ```

    """
    _UID_PREFIXES.setdefault(prefix, 0)
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]

def serialize_object(instance):
    if instance is None:
        return None
    if hasattr(instance, 'get_config'):
        return {
            'class_name': instance.__class__.__name__,
            'config': instance.get_config()
        }
    if hasattr(instance, '__name__'):
        return instance.__name__
    else:
        raise ValueError('Cannot serialize', instance)

def deserialize_object(identifier, module_objects=None,
                             custom_objects=None,
                             printable_module_name='object'):
    if isinstance(identifier, dict):
        # In this case we are dealing with a Keras config dictionary.
        config = identifier
        if 'class_name' not in config or 'config' not in config:
            raise ValueError('Improper config format: ' + str(config))
        class_name = config['class_name']
        if custom_objects and class_name in custom_objects:
            cls = custom_objects[class_name]
        elif class_name in _GLOBAL_CUSTOM_OBJECTS:
            cls = _GLOBAL_CUSTOM_OBJECTS[class_name]
        else:
            module_objects = module_objects or {}
            cls = module_objects.get(class_name)
            if cls is None:
                raise ValueError('Unknown ' + printable_module_name +
                                 ': ' + class_name)
        if hasattr(cls, 'from_config'):
            arg_spec = inspect.getargspec(cls.from_config)
            if 'custom_objects' in arg_spec.args:
                custom_objects = custom_objects or {}
                return cls.from_config(config['config'],
                                       custom_objects=dict(list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                                                           list(custom_objects.items())))
            return cls.from_config(config['config'])
        else:
            # Then `cls` may be a function returning a class.
            # in this case by convention `config` holds
            # the kwargs of the function.
            return cls(**config['config'])
    elif isinstance(identifier, six.string_types):
        function_name = identifier
        if custom_objects and function_name in custom_objects:
            fn = custom_objects.get(function_name)
        elif function_name in _GLOBAL_CUSTOM_OBJECTS:
            fn = _GLOBAL_CUSTOM_OBJECTS[function_name]
        else:
            fn = module_objects.get(function_name)
            if fn is None:
                raise ValueError('Unknown ' + printable_module_name,
                                 ':' + function_name)
        return fn
    else:
        raise ValueError('Could not interpret serialized ' +
                         printable_module_name + ': ' + identifier)