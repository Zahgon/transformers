
import html

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends


if is_bs4_available():
    import bs4
    from bs4 import BeautifulSoup


logger = logging.get_logger(__name__)


class MarkupLMFeatureExtractor(FeatureExtractionMixin):

    def __init__(self, **kwargs):
        requires_backends(self, ["bs4"])
        super().__init__(**kwargs)

    def xpath_soup(self, element):
        pass

    def get_three_from_single(self, html_string):
        pass

    def construct_xpath(self, xpath_tags, xpath_subscripts):
        pass

    def __call__(self, html_strings) -> BatchFeature:
        """
        Main method to prepare for the model one or several HTML strings.

        Args:
            html_strings (`str`, `list[str]`):
                The HTML string or batch of HTML strings from which to extract nodes and corresponding xpaths.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **nodes** -- Nodes.
            - **xpaths** -- Corresponding xpaths.

        Examples:

        ```python
        >>> from transformers import MarkupLMFeatureExtractor

        >>> page_name_1 = "page1.html"
        >>> page_name_2 = "page2.html"
        >>> page_name_3 = "page3.html"

        >>> with open(page_name_1) as f:
        ...     single_html_string = f.read()

        >>> feature_extractor = MarkupLMFeatureExtractor()

        >>> # single example
        >>> encoding = feature_extractor(single_html_string)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])

        >>> # batched example

        >>> multi_html_strings = []

        >>> with open(page_name_2) as f:
        ...     multi_html_strings.append(f.read())
        >>> with open(page_name_3) as f:
        ...     multi_html_strings.append(f.read())

        >>> encoding = feature_extractor(multi_html_strings)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])
        ```"""

        valid_strings = False

        if isinstance(html_strings, str):
            valid_strings = True
        elif isinstance(html_strings, (list, tuple)):
            if len(html_strings) == 0 or isinstance(html_strings[0], str):
                valid_strings = True

        if not valid_strings:
            raise ValueError(
                "HTML strings must of type `str`, `list[str]` (batch of examples), "
                f"but is of type {type(html_strings)}."
            )

        is_batched = isinstance(html_strings, (list, tuple)) and (isinstance(html_strings[0], str))

        if not is_batched:
            html_strings = [html_strings]

        nodes = []
        xpaths = []
        for html_string in html_strings:
            all_doc_strings, string2xtag_seq, string2xsubs_seq = self.get_three_from_single(html_string)
            nodes.append(all_doc_strings)
            xpath_strings = []
            for node, tag_list, sub_list in zip(all_doc_strings, string2xtag_seq, string2xsubs_seq):
                xpath_string = self.construct_xpath(tag_list, sub_list)
                xpath_strings.append(xpath_string)
            xpaths.append(xpath_strings)

        data = {"nodes": nodes, "xpaths": xpaths}
        encoded_inputs = BatchFeature(data=data, tensor_type=None)

        return encoded_inputs


__all__ = ["MarkupLMFeatureExtractor"]
