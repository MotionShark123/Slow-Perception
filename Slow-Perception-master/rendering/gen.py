import os
from pdf2image import convert_from_path


def edit_tex_file(new_tex_path, json_content):
        content = f'''
            \\documentclass{{standalone}}
            \\usepackage{{tikz}}
            \\usetikzlibrary{{angles, quotes}}
            \\usepackage{{tkz-euclide}}
            \\begin{{document}}
            \\begin{{tikzpicture}}
            {json_content}
            \\end{{tikzpicture}}
            \\end{{document}}
        '''
        with open(new_tex_path, 'w') as file:
            file.writelines(content)

def compile_tex_to_pdf(tex_path):
    os.system(f'xelatex -interaction=batchmode {tex_path} > curr.log')

def convert_pdf_to_png(pdf_path, output_folder, index, prefix=""):
    images = convert_from_path(pdf_path, dpi=256)
    images[0].save(f'{output_folder}/{prefix}page_{index}.png', 'PNG')
