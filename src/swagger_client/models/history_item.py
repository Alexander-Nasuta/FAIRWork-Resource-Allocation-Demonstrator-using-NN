# coding: utf-8

"""
    Api Documentation

    Api Documentation  # noqa: E501

    OpenAPI spec version: 1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class HistoryItem(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'date_time': 'str',
        'id': 'str',
        'name': 'str',
        'operation': 'str',
        'phase': 'str',
        'type_name': 'str',
        'user_name': 'str',
        'version_tag': 'str'
    }

    attribute_map = {
        'date_time': 'date_time',
        'id': 'id',
        'name': 'name',
        'operation': 'operation',
        'phase': 'phase',
        'type_name': 'type_name',
        'user_name': 'user_name',
        'version_tag': 'version_tag'
    }

    def __init__(self, date_time=None, id=None, name=None, operation=None, phase=None, type_name=None, user_name=None, version_tag=None):  # noqa: E501
        """HistoryItem - a model defined in Swagger"""  # noqa: E501
        self._date_time = None
        self._id = None
        self._name = None
        self._operation = None
        self._phase = None
        self._type_name = None
        self._user_name = None
        self._version_tag = None
        self.discriminator = None
        if date_time is not None:
            self.date_time = date_time
        if id is not None:
            self.id = id
        if name is not None:
            self.name = name
        if operation is not None:
            self.operation = operation
        if phase is not None:
            self.phase = phase
        if type_name is not None:
            self.type_name = type_name
        if user_name is not None:
            self.user_name = user_name
        if version_tag is not None:
            self.version_tag = version_tag

    @property
    def date_time(self):
        """Gets the date_time of this HistoryItem.  # noqa: E501


        :return: The date_time of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._date_time

    @date_time.setter
    def date_time(self, date_time):
        """Sets the date_time of this HistoryItem.


        :param date_time: The date_time of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._date_time = date_time

    @property
    def id(self):
        """Gets the id of this HistoryItem.  # noqa: E501


        :return: The id of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this HistoryItem.


        :param id: The id of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def name(self):
        """Gets the name of this HistoryItem.  # noqa: E501


        :return: The name of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this HistoryItem.


        :param name: The name of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def operation(self):
        """Gets the operation of this HistoryItem.  # noqa: E501


        :return: The operation of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._operation

    @operation.setter
    def operation(self, operation):
        """Sets the operation of this HistoryItem.


        :param operation: The operation of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._operation = operation

    @property
    def phase(self):
        """Gets the phase of this HistoryItem.  # noqa: E501


        :return: The phase of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._phase

    @phase.setter
    def phase(self, phase):
        """Sets the phase of this HistoryItem.


        :param phase: The phase of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._phase = phase

    @property
    def type_name(self):
        """Gets the type_name of this HistoryItem.  # noqa: E501


        :return: The type_name of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._type_name

    @type_name.setter
    def type_name(self, type_name):
        """Sets the type_name of this HistoryItem.


        :param type_name: The type_name of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._type_name = type_name

    @property
    def user_name(self):
        """Gets the user_name of this HistoryItem.  # noqa: E501


        :return: The user_name of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._user_name

    @user_name.setter
    def user_name(self, user_name):
        """Sets the user_name of this HistoryItem.


        :param user_name: The user_name of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._user_name = user_name

    @property
    def version_tag(self):
        """Gets the version_tag of this HistoryItem.  # noqa: E501


        :return: The version_tag of this HistoryItem.  # noqa: E501
        :rtype: str
        """
        return self._version_tag

    @version_tag.setter
    def version_tag(self, version_tag):
        """Sets the version_tag of this HistoryItem.


        :param version_tag: The version_tag of this HistoryItem.  # noqa: E501
        :type: str
        """

        self._version_tag = version_tag

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(HistoryItem, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, HistoryItem):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
