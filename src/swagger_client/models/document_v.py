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

class DocumentV(object):
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
        'approver': 'str',
        'assignment_id': 'int',
        'assignment_role': 'str',
        'content_id': 'str',
        'created': 'str',
        'created_by': 'str',
        'definition_id': 'int',
        'description': 'str',
        'discipline': 'str',
        'doc_id': 'int',
        'external_version': 'str',
        'format_id': 'str',
        'id': 'str',
        'original_name': 'str',
        'project_phase': 'str',
        'release_manager': 'str',
        'representing_id': 'int',
        'responsible': 'str',
        'reviewer': 'str',
        'rid': 'str',
        'sent_by': 'str',
        'sent_to': 'str',
        'source': 'str',
        'status': 'str',
        'title': 'str',
        'ver': 'str',
        'version_id': 'int'
    }

    attribute_map = {
        'approver': 'approver',
        'assignment_id': 'assignment_id',
        'assignment_role': 'assignment_role',
        'content_id': 'content_id',
        'created': 'created',
        'created_by': 'created_by',
        'definition_id': 'definition_id',
        'description': 'description',
        'discipline': 'discipline',
        'doc_id': 'doc_id',
        'external_version': 'external_version',
        'format_id': 'format_id',
        'id': 'id',
        'original_name': 'original_name',
        'project_phase': 'project_phase',
        'release_manager': 'release_manager',
        'representing_id': 'representing_id',
        'responsible': 'responsible',
        'reviewer': 'reviewer',
        'rid': 'rid',
        'sent_by': 'sent_by',
        'sent_to': 'sent_to',
        'source': 'source',
        'status': 'status',
        'title': 'title',
        'ver': 'ver',
        'version_id': 'version_id'
    }

    def __init__(self, approver=None, assignment_id=None, assignment_role=None, content_id=None, created=None, created_by=None, definition_id=None, description=None, discipline=None, doc_id=None, external_version=None, format_id=None, id=None, original_name=None, project_phase=None, release_manager=None, representing_id=None, responsible=None, reviewer=None, rid=None, sent_by=None, sent_to=None, source=None, status=None, title=None, ver=None, version_id=None):  # noqa: E501
        """DocumentV - a model defined in Swagger"""  # noqa: E501
        self._approver = None
        self._assignment_id = None
        self._assignment_role = None
        self._content_id = None
        self._created = None
        self._created_by = None
        self._definition_id = None
        self._description = None
        self._discipline = None
        self._doc_id = None
        self._external_version = None
        self._format_id = None
        self._id = None
        self._original_name = None
        self._project_phase = None
        self._release_manager = None
        self._representing_id = None
        self._responsible = None
        self._reviewer = None
        self._rid = None
        self._sent_by = None
        self._sent_to = None
        self._source = None
        self._status = None
        self._title = None
        self._ver = None
        self._version_id = None
        self.discriminator = None
        if approver is not None:
            self.approver = approver
        if assignment_id is not None:
            self.assignment_id = assignment_id
        if assignment_role is not None:
            self.assignment_role = assignment_role
        if content_id is not None:
            self.content_id = content_id
        if created is not None:
            self.created = created
        if created_by is not None:
            self.created_by = created_by
        if definition_id is not None:
            self.definition_id = definition_id
        if description is not None:
            self.description = description
        if discipline is not None:
            self.discipline = discipline
        if doc_id is not None:
            self.doc_id = doc_id
        if external_version is not None:
            self.external_version = external_version
        if format_id is not None:
            self.format_id = format_id
        if id is not None:
            self.id = id
        if original_name is not None:
            self.original_name = original_name
        if project_phase is not None:
            self.project_phase = project_phase
        if release_manager is not None:
            self.release_manager = release_manager
        if representing_id is not None:
            self.representing_id = representing_id
        if responsible is not None:
            self.responsible = responsible
        if reviewer is not None:
            self.reviewer = reviewer
        if rid is not None:
            self.rid = rid
        if sent_by is not None:
            self.sent_by = sent_by
        if sent_to is not None:
            self.sent_to = sent_to
        if source is not None:
            self.source = source
        if status is not None:
            self.status = status
        if title is not None:
            self.title = title
        if ver is not None:
            self.ver = ver
        if version_id is not None:
            self.version_id = version_id

    @property
    def approver(self):
        """Gets the approver of this DocumentV.  # noqa: E501


        :return: The approver of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._approver

    @approver.setter
    def approver(self, approver):
        """Sets the approver of this DocumentV.


        :param approver: The approver of this DocumentV.  # noqa: E501
        :type: str
        """

        self._approver = approver

    @property
    def assignment_id(self):
        """Gets the assignment_id of this DocumentV.  # noqa: E501


        :return: The assignment_id of this DocumentV.  # noqa: E501
        :rtype: int
        """
        return self._assignment_id

    @assignment_id.setter
    def assignment_id(self, assignment_id):
        """Sets the assignment_id of this DocumentV.


        :param assignment_id: The assignment_id of this DocumentV.  # noqa: E501
        :type: int
        """

        self._assignment_id = assignment_id

    @property
    def assignment_role(self):
        """Gets the assignment_role of this DocumentV.  # noqa: E501


        :return: The assignment_role of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._assignment_role

    @assignment_role.setter
    def assignment_role(self, assignment_role):
        """Sets the assignment_role of this DocumentV.


        :param assignment_role: The assignment_role of this DocumentV.  # noqa: E501
        :type: str
        """

        self._assignment_role = assignment_role

    @property
    def content_id(self):
        """Gets the content_id of this DocumentV.  # noqa: E501


        :return: The content_id of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._content_id

    @content_id.setter
    def content_id(self, content_id):
        """Sets the content_id of this DocumentV.


        :param content_id: The content_id of this DocumentV.  # noqa: E501
        :type: str
        """

        self._content_id = content_id

    @property
    def created(self):
        """Gets the created of this DocumentV.  # noqa: E501


        :return: The created of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._created

    @created.setter
    def created(self, created):
        """Sets the created of this DocumentV.


        :param created: The created of this DocumentV.  # noqa: E501
        :type: str
        """

        self._created = created

    @property
    def created_by(self):
        """Gets the created_by of this DocumentV.  # noqa: E501


        :return: The created_by of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """Sets the created_by of this DocumentV.


        :param created_by: The created_by of this DocumentV.  # noqa: E501
        :type: str
        """

        self._created_by = created_by

    @property
    def definition_id(self):
        """Gets the definition_id of this DocumentV.  # noqa: E501


        :return: The definition_id of this DocumentV.  # noqa: E501
        :rtype: int
        """
        return self._definition_id

    @definition_id.setter
    def definition_id(self, definition_id):
        """Sets the definition_id of this DocumentV.


        :param definition_id: The definition_id of this DocumentV.  # noqa: E501
        :type: int
        """

        self._definition_id = definition_id

    @property
    def description(self):
        """Gets the description of this DocumentV.  # noqa: E501


        :return: The description of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this DocumentV.


        :param description: The description of this DocumentV.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def discipline(self):
        """Gets the discipline of this DocumentV.  # noqa: E501


        :return: The discipline of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._discipline

    @discipline.setter
    def discipline(self, discipline):
        """Sets the discipline of this DocumentV.


        :param discipline: The discipline of this DocumentV.  # noqa: E501
        :type: str
        """

        self._discipline = discipline

    @property
    def doc_id(self):
        """Gets the doc_id of this DocumentV.  # noqa: E501


        :return: The doc_id of this DocumentV.  # noqa: E501
        :rtype: int
        """
        return self._doc_id

    @doc_id.setter
    def doc_id(self, doc_id):
        """Sets the doc_id of this DocumentV.


        :param doc_id: The doc_id of this DocumentV.  # noqa: E501
        :type: int
        """

        self._doc_id = doc_id

    @property
    def external_version(self):
        """Gets the external_version of this DocumentV.  # noqa: E501


        :return: The external_version of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._external_version

    @external_version.setter
    def external_version(self, external_version):
        """Sets the external_version of this DocumentV.


        :param external_version: The external_version of this DocumentV.  # noqa: E501
        :type: str
        """

        self._external_version = external_version

    @property
    def format_id(self):
        """Gets the format_id of this DocumentV.  # noqa: E501


        :return: The format_id of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._format_id

    @format_id.setter
    def format_id(self, format_id):
        """Sets the format_id of this DocumentV.


        :param format_id: The format_id of this DocumentV.  # noqa: E501
        :type: str
        """

        self._format_id = format_id

    @property
    def id(self):
        """Gets the id of this DocumentV.  # noqa: E501


        :return: The id of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this DocumentV.


        :param id: The id of this DocumentV.  # noqa: E501
        :type: str
        """

        self._id = id

    @property
    def original_name(self):
        """Gets the original_name of this DocumentV.  # noqa: E501


        :return: The original_name of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._original_name

    @original_name.setter
    def original_name(self, original_name):
        """Sets the original_name of this DocumentV.


        :param original_name: The original_name of this DocumentV.  # noqa: E501
        :type: str
        """

        self._original_name = original_name

    @property
    def project_phase(self):
        """Gets the project_phase of this DocumentV.  # noqa: E501


        :return: The project_phase of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._project_phase

    @project_phase.setter
    def project_phase(self, project_phase):
        """Sets the project_phase of this DocumentV.


        :param project_phase: The project_phase of this DocumentV.  # noqa: E501
        :type: str
        """

        self._project_phase = project_phase

    @property
    def release_manager(self):
        """Gets the release_manager of this DocumentV.  # noqa: E501


        :return: The release_manager of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._release_manager

    @release_manager.setter
    def release_manager(self, release_manager):
        """Sets the release_manager of this DocumentV.


        :param release_manager: The release_manager of this DocumentV.  # noqa: E501
        :type: str
        """

        self._release_manager = release_manager

    @property
    def representing_id(self):
        """Gets the representing_id of this DocumentV.  # noqa: E501


        :return: The representing_id of this DocumentV.  # noqa: E501
        :rtype: int
        """
        return self._representing_id

    @representing_id.setter
    def representing_id(self, representing_id):
        """Sets the representing_id of this DocumentV.


        :param representing_id: The representing_id of this DocumentV.  # noqa: E501
        :type: int
        """

        self._representing_id = representing_id

    @property
    def responsible(self):
        """Gets the responsible of this DocumentV.  # noqa: E501


        :return: The responsible of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._responsible

    @responsible.setter
    def responsible(self, responsible):
        """Sets the responsible of this DocumentV.


        :param responsible: The responsible of this DocumentV.  # noqa: E501
        :type: str
        """

        self._responsible = responsible

    @property
    def reviewer(self):
        """Gets the reviewer of this DocumentV.  # noqa: E501


        :return: The reviewer of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._reviewer

    @reviewer.setter
    def reviewer(self, reviewer):
        """Sets the reviewer of this DocumentV.


        :param reviewer: The reviewer of this DocumentV.  # noqa: E501
        :type: str
        """

        self._reviewer = reviewer

    @property
    def rid(self):
        """Gets the rid of this DocumentV.  # noqa: E501


        :return: The rid of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._rid

    @rid.setter
    def rid(self, rid):
        """Sets the rid of this DocumentV.


        :param rid: The rid of this DocumentV.  # noqa: E501
        :type: str
        """

        self._rid = rid

    @property
    def sent_by(self):
        """Gets the sent_by of this DocumentV.  # noqa: E501


        :return: The sent_by of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._sent_by

    @sent_by.setter
    def sent_by(self, sent_by):
        """Sets the sent_by of this DocumentV.


        :param sent_by: The sent_by of this DocumentV.  # noqa: E501
        :type: str
        """

        self._sent_by = sent_by

    @property
    def sent_to(self):
        """Gets the sent_to of this DocumentV.  # noqa: E501


        :return: The sent_to of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._sent_to

    @sent_to.setter
    def sent_to(self, sent_to):
        """Sets the sent_to of this DocumentV.


        :param sent_to: The sent_to of this DocumentV.  # noqa: E501
        :type: str
        """

        self._sent_to = sent_to

    @property
    def source(self):
        """Gets the source of this DocumentV.  # noqa: E501


        :return: The source of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._source

    @source.setter
    def source(self, source):
        """Sets the source of this DocumentV.


        :param source: The source of this DocumentV.  # noqa: E501
        :type: str
        """

        self._source = source

    @property
    def status(self):
        """Gets the status of this DocumentV.  # noqa: E501


        :return: The status of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this DocumentV.


        :param status: The status of this DocumentV.  # noqa: E501
        :type: str
        """

        self._status = status

    @property
    def title(self):
        """Gets the title of this DocumentV.  # noqa: E501


        :return: The title of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this DocumentV.


        :param title: The title of this DocumentV.  # noqa: E501
        :type: str
        """

        self._title = title

    @property
    def ver(self):
        """Gets the ver of this DocumentV.  # noqa: E501


        :return: The ver of this DocumentV.  # noqa: E501
        :rtype: str
        """
        return self._ver

    @ver.setter
    def ver(self, ver):
        """Sets the ver of this DocumentV.


        :param ver: The ver of this DocumentV.  # noqa: E501
        :type: str
        """

        self._ver = ver

    @property
    def version_id(self):
        """Gets the version_id of this DocumentV.  # noqa: E501


        :return: The version_id of this DocumentV.  # noqa: E501
        :rtype: int
        """
        return self._version_id

    @version_id.setter
    def version_id(self, version_id):
        """Sets the version_id of this DocumentV.


        :param version_id: The version_id of this DocumentV.  # noqa: E501
        :type: int
        """

        self._version_id = version_id

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
        if issubclass(DocumentV, dict):
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
        if not isinstance(other, DocumentV):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
