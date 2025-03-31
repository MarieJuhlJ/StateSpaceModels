class LayerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_cls):
            cls._registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown layer type: {name}")
        return cls._registry[name](*args, **kwargs)

class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(layer_cls):
            cls._registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def create(cls, cfg):
        if cfg.name not in cls._registry:
            raise ValueError(f"Unknown layer type: {cfg.name}")
        return cls._registry[cfg.name](cfg)