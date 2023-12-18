import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

import torch
from utils import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor, word_to_image

forward_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main(det_archs, reco_archs):
    
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("Delivery Order OCR")

    st.caption("## Text Detection by ResNet Family")
    # Set the columns
    cols = st.columns((1, 1))
    cols[0].subheader("Input page")
    cols[1].subheader("Segmentation heatmap")
    
    # Sidebar
    # File selection
    st.sidebar.title("Document selection")
    # Disabling warning
    # st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
        page = doc[page_idx]
        cols[0].image(page)

    # Model selection
    st.sidebar.title("Model selection")
    det_arch = st.sidebar.selectbox("Text detection model", det_archs)
    reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

    # For newline
    st.sidebar.write("\n")

    # Binarization threshold
    bin_thresh = st.sidebar.slider("Binarization threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    st.sidebar.write("\n")
    
    st.sidebar.title("Explanation selection")
    
    discard_ratio = st.sidebar.slider("Dicard Ratio", min_value=0.0, max_value=0.9, value=0.9, step=0.1)
    head_fusion = st.sidebar.selectbox("Head Fusion", ["min", "max", "mean"])

    if st.sidebar.button("Analyze page"):
        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner("Loading model..."):
                predictor = load_predictor(
                    det_arch, reco_arch, True, False, bin_thresh, head_fusion, discard_ratio, forward_device
                )

            with st.spinner("Analyzing..."):
                # Forward the image to the model
                seg_map_ = forward_image(predictor, page, forward_device)
                seg_map = np.squeeze(seg_map_)
                seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(seg_map)
                ax.axis("off")
                cols[1].pyplot(fig)

                st.markdown("\n---\n")

                st.caption("## Text Recognition by (VitSTR) VisionTransformer for Scene Text Recognition")
                ocr_cols = st.columns((1, 1))
                ocr_cols[0].subheader("OCR output")
                ocr_cols[1].subheader("Aligned recognition result")


                # Plot OCR output
                out = predictor([page])
                fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)
                ocr_cols[0].pyplot(fig)

                # Page reconsitution under input page
                page_export = out.pages[0].export()

                all_words = []
                for block in page_export["blocks"]:
                    for line in block["lines"]:
                        for word in line["words"]:
                            all_words.append(word)
        
                # sort by y1
                all_words.sort(key=lambda x: x["geometry"][0][1])
                # group by y1
                lines = []
                line = []
                for word in all_words:
                    if len(line) == 0:
                        line.append(word)
                    else:
                        last = line[-1]["geometry"]
                        curr = word["geometry"]

                        # Overlap starts
                        if last[0][1] <= curr[0][1] <= last[1][1]:
                            common = max(0, min(last[1][1], curr[1][1]) - max(last[0][1], curr[0][1]))
                            overlap = common / (last[1][1] - last[0][1] + curr[1][1] - curr[0][1] - common)
                            if overlap > 0.4:
                                line.append(word)
                            else:
                                lines.append(line)
                                line = [word]
                        else:
                            lines.append(line)
                            line = [word]
                lines.append(line)
                        
                # sort by x1
                for line in lines:
                    line.sort(key=lambda x: x["geometry"][0][0])

                for line in lines:
                    for word in line:
                        image = word_to_image(word, page)
                        word["image"] = image
                
                texts = []
                for line in lines:
                    text = ""
                    last_word = None
                    for word in line:
                        # print(word["value"], word["geometry"])
                        if last_word is not None:
                            if last_word["geometry"][1][0] >= word["geometry"][0][0]:
                                text += " "
                            else:
                                text += "\t"
                        text += word["value"]
                        last_word = word
                    texts.append(text)
                result = "\n".join(texts)
                ocr_cols[1].code(result, language="text", line_numbers=True)

                st.markdown("\n---\n")
                st.markdown("## Attention Rollout Explanation")
                
                exp = predictor.explanations
                length = len(exp)
                bottom_columns = st.columns((length, length))
                exp = sorted(exp, key=lambda x: x[1], reverse=False)
                
                for i, row in enumerate(exp):
                    image = row[2]
                    image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
                    mask = cv2.resize(row[3], (image.shape[1], image.shape[0]))
                    mask = mask = np.uint8(255 * mask)
                    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_RAINBOW)
                    heatmap = np.float32(heatmap) / 255
                    cam = heatmap + np.float32(image)
                    cam = cam / np.max(cam)
                    cam = np.uint8(255 * cam)
                    bottom_columns[0].image(image, clamp=True)
                    bottom_columns[0].markdown(f'- {row[0]}')
                    bottom_columns[1].image(cam, clamp=True)
                    bottom_columns[1].markdown(f'- {row[1]:.2f}')

if __name__ == "__main__":
    main(DET_ARCHS, RECO_ARCHS)
