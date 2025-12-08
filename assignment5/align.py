#!/usr/bin/env python3
import argparse
import os
import xml.etree.ElementTree as ET

def main():
    parser = argparse.ArgumentParser(description="Extract parallel sentences from a TMX file and write aligned files.")
    parser.add_argument("tmx_file", help="Path to the TMX file.")
    parser.add_argument("output_dir", help="Directory to write aligned files.")
    args = parser.parse_args()

    tmx_file = args.tmx_file
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Parse TMX and find the first two language codes
    tree = ET.parse(tmx_file)
    root = tree.getroot()

    codes = []
    # xml namespace attribute for xml:lang
    lang_attrs = ("{http://www.w3.org/XML/1998/namespace}lang", "xml:lang", "lang")
    for tuv in root.findall(".//tuv"):
        lang = None
        for attr in lang_attrs:
            lang = tuv.attrib.get(attr)
            if lang:
                break
        if lang and lang not in codes:
            codes.append(lang)
            if len(codes) >= 2:
                break

    if len(codes) < 2:
        print("Error: TMX file contains fewer than 2 languages.")
        return

    code1, code2 = codes
    file1_path = os.path.join(output_dir, f"aligned_{code1}.{code2}")
    file2_path = os.path.join(output_dir, f"aligned_{code2}.{code1}")

    with open(file1_path, "w", encoding="utf-8") as f1, \
         open(file2_path, "w", encoding="utf-8") as f2:

        count = 0
        # Iterate over translation units
        for tu in root.findall(".//tu"):
            if count >= 5000:
                break

            texts = {}
            for tuv in tu.findall("tuv"):
                lang = None
                for attr in lang_attrs:
                    lang = tuv.attrib.get(attr)
                    if lang:
                        break
                if lang in (code1, code2):
                    seg = tuv.find("seg")
                    if seg is not None and seg.text:
                        texts[lang] = seg.text.strip()

            if code1 in texts and code2 in texts:
                f1.write(texts[code1].replace("\n", " ") + "\n")
                f2.write(texts[code2].replace("\n", " ") + "\n")
                count += 1

if __name__ == "__main__":
    main()
