import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from llama_index.core.schema import ImageNode, TextNode
import re


class CustomTreeProcessor(Treeprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = []

    def run(self, root):
        current_text = []

        for element in root:
            if element.tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_text:
                    self.nodes.append(TextNode(text=''.join(current_text).strip()))
                    current_text = []

            elif element.tag == 'p':
                if element.text is not None:
                    text = re.sub(r'\\(.)', r'\1', element.text)
                    current_text.append(text)

                for child in element.iter():
                    if child.tag == 'img':
                        # src = child.get('src')
                        # self.nodes.append(ImageNode(text="this is an image", image_url=src))
                        pass
                    elif child.text is not None:
                        current_text.append(child.text)

            elif element.tag in ['ul', 'ol']:
                for li in element.findall('li'):
                    if li.text is not None:
                        list_item_text = re.sub(r'\\(.)', r'\1', li.text)
                        current_text.append(f"- {list_item_text}\n")  

            elif element.tag == 'img':
                # src = element.get('src')
                # self.nodes.append(ImageNode(text="this is an image", image_url=src))
                pass

        if current_text:
            self.nodes.append(TextNode(text=''.join(current_text).strip()))

        return root


class CustomExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(CustomTreeProcessor(md), 'custom_tree_processor', 15)


def process_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
    md = markdown.Markdown(extensions=[CustomExtension()])
    md.convert(md_text)
    return md.treeprocessors['custom_tree_processor'].nodes


def remove_json_markers(text):
    if text.startswith('```json'):
        text = text[len('```json'):]
    if text.endswith('```'):
        text = text[:-len('```')]
    return text.strip()


def unescape_markdown(text):
    """
    Convert escape characters in Markdown text to their actual meaning.
    For example, convert `\\n` to a newline, `\\t` to a tab, etc.
    """
    if isinstance(text, str):
        return text.encode().decode('unicode_escape')
    return text