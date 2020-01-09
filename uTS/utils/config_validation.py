def validate_attributes(default_attributes, kwargs):
    """
    Validate the keys from kwargs are well defined, all within the attributes of default_attributes;
    The attributes could be nested, therefore inside a recursive in the loop.
    :param default_attributes: dict
            Predefined dictionary of a class __init__() optional variables, in the format of (
            attribute, value)
    :param kwargs: dict
            User input of a class __init__() optional variables, in the format of (attributes,
            default value)
    :return: Nothing
    """
    if not set(kwargs.keys()).issubset(set(default_attributes.keys())):
        raise NameError(
            "Invalid keys: {} and we only accept the following variables {}".format(
                list(set(kwargs.keys()).difference(set(default_attributes.keys()))),
                default_attributes.keys(),
            ),
        )

    for key, value in kwargs.iteritems():
        if isinstance(value, dict):
            validate_attributes(default_attributes.get(key), value)


def update_attributes_with_default(p_object, default_attributes, kwargs, prefix=""):
    """
    Comparing a user defined dictionary (could be nested) with a default fixed dictionary with
    format
    of (attribute, default value),
    if user doesn't provide the key, set p_object.attribute_key = default_value; if user does
    provide the key and
    a value, set p_object.attribute_key = user_defined_value.
    :param p_object: object
            The object needs to be modified for all of the attributes, based on either
            default_attributes or kwargs.
    :param default_attributes: dict
            Predefined dictionary of a class __init__() optional variables, in the format of (
            attribute, value)
    :param prefix: str
            The prefix string of the attribute name, so it has the format of a.b.c.
    :param kwargs: dict
            User input of a class __init__() optional variables, in the format of (attributes,
            default value)
    :return: Nothing
    """
    for attr, default in default_attributes.iteritems():
        new_attr = "{}_{}".format(prefix, attr) if prefix else attr
        setattr(p_object, new_attr, kwargs.get(attr, default))
        if isinstance(default, dict):
            update_attributes_with_default(
                p_object, default, kwargs.get(
                    attr, default,
                ), prefix=new_attr,
            )


def validate_and_setup_input_list(input_list, num_expected, default_value):
    """
    For the given input_list, if it is not None nor empty, check it whether has the expected
    length; otherwise,
    return an array of length num_expected and value=default_value
    :param input_list: array
            Given list, check whether it has valid length if it is not None nor empty.
    :param num_expected: int
            Expected array length
    :param default_value: python basic data types, str/int/float/...
            The default value of the return array if array is None or empty
    :return:
    """
    if input_list is None or len(input_list) == 0:
        input_list = [default_value] * num_expected
    else:
        assert num_expected == len(input_list)
    return input_list
