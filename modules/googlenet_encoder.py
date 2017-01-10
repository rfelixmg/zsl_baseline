def googlenet_encoder():
    from googlenet.googlenet_custom_layers import PoolHelper,LRN
    from keras.models import model_from_json
    import sys, os

    directory, _ = os.path.split(os.path.abspath(__file__))

    model = model_from_json(open(directory + '/googlenet/googlenet_architecture.json').read(),
                            custom_objects={"PoolHelper": PoolHelper,"LRN":LRN})
    model.load_weights( directory + '/googlenet/googlenet_weights.h5')

    return model