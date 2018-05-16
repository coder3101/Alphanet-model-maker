import warnings


def validate_config(dataMap):
    try:
        if not isinstance(dataMap['dataset']['single-data-source'], bool):
            raise ValueError('Required value for single-data-source is', bool)

        is_single_source = dataMap['dataset']['single-data-source']

        if str(dataMap['dataset']['unified-data-location']) == "None" and is_single_source:
            raise ValueError(
                'unified-data-location is a required field and needs', str)

        if is_single_source and str(dataMap['dataset']['train-data-location']) != "None":
            warnings.warn(
                'Using unified dataset location skipping values of train and test')

        if str(dataMap['dataset']['train-data-location']) == "None" and not is_single_source:
            raise ValueError(
                'train-data-location is a required field and needs ', str)

        if str(dataMap['dataset']['test-data-location']) == "None" and not is_single_source:
            raise ValueError(
                'test-data-location is a required field and needs ', str)

        if str(dataMap['dataset']['data-format']) != "csv" and str(dataMap['dataset']['data-format']) != "binary":
            raise ValueError('data-format can only be one of',
                             ['csv', 'binary'])

        # Validation for model starts here

        rootMap = dataMap['model']
        if not isinstance(rootMap['learning_rate'], float) or rootMap['learning_rate'] > 100:
            raise ValueError(
                'Invalid value for learning_rate. Requires ', float, 'less than 100')

        if str(rootMap['type']) != "feed_forward" and str(rootMap['type']) != "convolution":
            raise ValueError('type can only be one of',
                             ['feed_forward', 'convolution'])

        if str(rootMap['name']) is None:
            raise ValueError('Name for a model is missing from yaml')

        if not isinstance(rootMap['layer_num'], int) or rootMap['layer_num'] > 100:
            raise ValueError(
                'layer_num should be less than 100 and a valid', int)

        if not isinstance(rootMap['batch_size'], int):
            raise ValueError('Batch size must be a int')

        for layer_dim in rootMap['layer_dims'].split(','):
            try:
                layer_dim = int(layer_dim)
                if layer_dim <= 0:
                    raise ValueError('Only values more than 0 are accepted')
            except ValueError as e:
                raise ValueError(
                    'Layer dims got unexpected value, required only positive numbers', e)

    except KeyError as e:
        print('You modified the yaml keys or invalid config.yaml', e)
