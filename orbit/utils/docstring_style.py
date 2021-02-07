from __future__ import absolute_import

from collections import OrderedDict
from inspect import cleandoc
from custom_inherit._doc_parse_tools.numpy_parse_tools import \
        parse_numpy_doc

__all__ = ["merge_numpy_docs"]


def merge_section(key, prnt_sec, child_sec, merge_within_sections=True):
    """ Synthesize a output numpy docstring section.
    Parameters
    ----------
    key: str
        The numpy-section being merged.
    prnt_sec: Optional[str]
        The docstring section from the parent's attribute.
    child_sec: Optional[str]
        The docstring section from the child's attribute.
    merge_within_sections: bool
        merge parents and child docstring
    Returns
    -------
    Optional[str]
        The output docstring section.
    """

    doc_sections_that_cant_merge = [
        "Short Summary",
        "Deprecation Warning",
        "Extended Summary",
        "Examples"
    ]

    def common_start(sa, sb):
        """ returns the longest common substring from the beginning of sa and sb """
        def _iter():
            for a, b in zip(sa, sb):
                if a == b:
                    yield a
                else:
                    return

        return ''.join(_iter())

    if prnt_sec is None and child_sec is None:
        return None

    if key == "Short Summary":
        header = ""
    else:
        header = "\n".join((key, "".join("-" for i in range(len(key))), ""))

    if merge_within_sections and key not in doc_sections_that_cant_merge:
        if child_sec is None:
            body = prnt_sec
        elif prnt_sec is None:
            body = child_sec
        else:
            # only add same portion once
            common = common_start(prnt_sec, child_sec)
            n = len(common)
            # remove additional 'new line' when append child
            n = n + 1 if child_sec[n:].startswith('\n') else n
            # ensure the common substring to be at least 10 character long
            n = 0 if  n < 10 else n
            body = '\n'.join((prnt_sec, child_sec[n:]))
    else:
        body = prnt_sec if child_sec is None else child_sec

    return header + body


def merge_all_sections(prnt_sctns, child_sctns, merge_within_sections=True):
    """ Merge the doc-sections of the parent's and child's attribute into a single docstring.
    Parameters
    ----------
    prnt_sctns: OrderedDict[str, Union[None,str]]
    child_sctns: OrderedDict[str, Union[None,str]]
    merge_within_sections: bool
        merge parents and child docstring
    Returns
    -------
    str
        Output docstring of the merged docstrings.
    """
    doc = []

    prnt_only_raises = prnt_sctns["Raises"] and not (
        prnt_sctns["Returns"] or prnt_sctns["Yields"]
    )
    if prnt_only_raises and (child_sctns["Returns"] or child_sctns["Yields"]):
        prnt_sctns["Raises"] = None

    for key in prnt_sctns:
        sect = merge_section(
            key,
            prnt_sctns[key],
            child_sctns[key],
            merge_within_sections=merge_within_sections
        )
        if sect is not None:
            doc.append(sect)
    return "\n\n".join(doc) if doc else None


def merge_numpy_docs_dedup(prnt_doc=None, child_doc=None, merge_within_sections=True):
    """ Merge two numpy-style docstrings into a single docstring.
    Given the numpy-style docstrings from a parent and child's attributes, merge the docstring
    sections such that the child's section is used, wherever present, otherwise the parent's
    section is used.
    Any whitespace that can be uniformly removed from a docstring's second line and onwards is
    removed. Sections will be separated by a single blank line.
    Parameters
    ----------
    prnt_doc: Optional[str]
        The docstring from the parent.
    child_doc: Optional[str]
        The docstring from the child.
    merge_within_sections: bool
        merge parents and child docstring
    Returns
    -------
    Union[str, None]
        The merged docstring.
    """
    return merge_all_sections(
        parse_numpy_doc(prnt_doc),
        parse_numpy_doc(child_doc),
        merge_within_sections=merge_within_sections
    )
