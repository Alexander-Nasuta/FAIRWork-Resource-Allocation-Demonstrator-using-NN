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

class PropInfo(object):
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
        'domain': 'str',
        'editable': 'bool',
        'in_use': 'bool',
        'urn': 'str'
    }

    attribute_map = {
        'domain': 'domain',
        'editable': 'editable',
        'in_use': 'inUse',
        'urn': 'urn'
    }

    def __init__(self, domain=None, editable=None, in_use=None, urn=None):  # noqa: E501
        """PropInfo - a model defined in Swagger"""  # noqa: E501
        self._domain = None
        self._editable = None
        self._in_use = None
        self._urn = None
        self.discriminator = None
        if domain is not None:
            self.domain = domain
        if editable is not None:
            self.editable = editable
        if in_use is not None:
            self.in_use = in_use
        if urn is not None:
            self.urn = urn

    @property
    def domain(self):
        """Gets the domain of this PropInfo.  # noqa: E501


        :return: The domain of this PropInfo.  # noqa: E501
        :rtype: str
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        """Sets the domain of this PropInfo.


        :param domain: The domain of this PropInfo.  # noqa: E501
        :type: str
        """

        self._domain = domain

    @property
    def editable(self):
        """Gets the editable of this PropInfo.  # noqa: E501


        :return: The editable of this PropInfo.  # noqa: E501
        :rtype: bool
        """
        return self._editable

    @editable.setter
    def editable(self, editable):
        """Sets the editable of this PropInfo.


        :param editable: The editable of this PropInfo.  # noqa: E501
        :type: bool
        """

        self._editable = editable

    @property
    def in_use(self):
        """Gets the in_use of this PropInfo.  # noqa: E501


        :return: The in_use of this PropInfo.  # noqa: E501
        :rtype: bool
        """
        return self._in_use

    @in_use.setter
    def in_use(self, in_use):
        """Sets the in_use of this PropInfo.


        :param in_use: The in_use of this PropInfo.  # noqa: E501
        :type: bool
        """

        self._in_use = in_use

    @property
    def urn(self):
        """Gets the urn of this PropInfo.  # noqa: E501


        :return: The urn of this PropInfo.  # noqa: E501
        :rtype: str
        """
        return self._urn

    @urn.setter
    def urn(self, urn):
        """Sets the urn of this PropInfo.


        :param urn: The urn of this PropInfo.  # noqa: E501
        :type: str
        """

        self._urn = urn

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
        if issubclass(PropInfo, dict):
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
        if not isinstance(other, PropInfo):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other