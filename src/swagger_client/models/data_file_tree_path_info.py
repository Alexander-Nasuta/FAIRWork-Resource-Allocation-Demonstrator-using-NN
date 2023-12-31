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

class DataFileTreePathInfo(object):
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
        'data_type': 'str',
        'description': 'str',
        'file_name': 'str',
        'id': 'str',
        'instance_id': 'int',
        'title': 'str',
        'tree_path_to_linked_bkdn_elem': 'list[int]'
    }

    attribute_map = {
        'data_type': 'data_type',
        'description': 'description',
        'file_name': 'file_name',
        'id': 'id',
        'instance_id': 'instance_id',
        'title': 'title',
        'tree_path_to_linked_bkdn_elem': 'tree_path_to_linked_bkdn_elem'
    }

    def __init__(self, data_type=None, description=None, file_name=None, id=None, instance_id=None, title=None, tree_path_to_linked_bkdn_elem=None):  # noqa: E501
        """DataFileTreePathInfo - a model defined in Swagger"""  # noqa: E501
        self._data_type = None
        self._description = None
        self._file_name = None
        self._id = None
        self._instance_id = None
        self._title = None
        self._tree_path_to_linked_bkdn_elem = None
        self.discriminator = None
        if data_type is not None:
            self.data_type = data_type
        if description is not None:
            self.description = description
        if file_name is not None:
            self.file_name = file_name
        if id is not None:
            self.id = id
        if instance_id is not None:
            self.instance_id = instance_id
        if title is not None:
            self.title = title
        if tree_path_to_linked_bkdn_elem is not None:
            self.tree_path_to_linked_bkdn_elem = tree_path_to_linked_bkdn_elem

    @property
    def data_type(self):
        """Gets the data_type of this DataFileTreePathInfo.  # noqa: E501


        :return: The data_type of this DataFileTreePathInfo.  # noqa: E501
        :rtype: str
        """
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        """Sets the data_type of this DataFileTreePathInfo.


        :param data_type: The data_type of this DataFileTreePathInfo.  # noqa: E501
        :type: str
        """

        self._data_type = data_type

    @property
    def description(self):
        """Gets the description of this DataFileTreePathInfo.  # noqa: E501


        :return: The description of this DataFileTreePathInfo.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this DataFileTreePathInfo.


        :param description: The description of this DataFileTreePathInfo.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def file_name(self):
        """Gets the file_name of this DataFileTreePathInfo.  # noqa: E501


        :return: The file_name of this DataFileTreePathInfo.  # noqa: E501
        :rtype: str
        """
        return self._file_name

    @file_name.setter
    def file_name(self, file_name):
        """Sets the file_name of this DataFileTreePathInfo.


        :param file_name: The file_name of this DataFileTreePathInfo.  # noqa: E501
        :type: str
        """

        self._file_name = file_name

    @property
    def id(self):
        """Gets the id of this DataFileTreePathInfo.  # noqa: E501


        :return: The id of this DataFileTreePathInfo.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this DataFileTreePathInfo.


        :param id: The id of this DataFileTreePathInfo.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def instance_id(self):
        """Gets the instance_id of this DataFileTreePathInfo.  # noqa: E501


        :return: The instance_id of this DataFileTreePathInfo.  # noqa: E501
        :rtype: int
        """
        return self._instance_id

    @instance_id.setter
    def instance_id(self, instance_id):
        """Sets the instance_id of this DataFileTreePathInfo.


        :param instance_id: The instance_id of this DataFileTreePathInfo.  # noqa: E501
        :type: int
        """

        self._instance_id = instance_id

    @property
    def title(self):
        """Gets the title of this DataFileTreePathInfo.  # noqa: E501


        :return: The title of this DataFileTreePathInfo.  # noqa: E501
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this DataFileTreePathInfo.


        :param title: The title of this DataFileTreePathInfo.  # noqa: E501
        :type: str
        """

        self._title = title

    @property
    def tree_path_to_linked_bkdn_elem(self):
        """Gets the tree_path_to_linked_bkdn_elem of this DataFileTreePathInfo.  # noqa: E501


        :return: The tree_path_to_linked_bkdn_elem of this DataFileTreePathInfo.  # noqa: E501
        :rtype: list[int]
        """
        return self._tree_path_to_linked_bkdn_elem

    @tree_path_to_linked_bkdn_elem.setter
    def tree_path_to_linked_bkdn_elem(self, tree_path_to_linked_bkdn_elem):
        """Sets the tree_path_to_linked_bkdn_elem of this DataFileTreePathInfo.


        :param tree_path_to_linked_bkdn_elem: The tree_path_to_linked_bkdn_elem of this DataFileTreePathInfo.  # noqa: E501
        :type: list[int]
        """

        self._tree_path_to_linked_bkdn_elem = tree_path_to_linked_bkdn_elem

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
        if issubclass(DataFileTreePathInfo, dict):
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
        if not isinstance(other, DataFileTreePathInfo):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
