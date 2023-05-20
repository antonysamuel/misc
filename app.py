import streamlit as st
import streamlit_drawable_canvas as st_canvas
from PIL import Image
import subprocess,os

def main():
    st.title("Image Processing with Streamlit")

    # File upload and display image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display the image with click support
        canvas = st_canvas.st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange color for keypoint
            stroke_width=8,
            stroke_color="rgba(255, 165, 0, 0.6)",
            background_color="#ffffff",
            background_image=image,
            height=image.size[1],
            width=image.size[0],
            drawing_mode="freedraw",
            key="canvas",
        )
        if st.button('print_k_pnts'):
            keypoint_data = canvas.json_data["objects"][0]
            print(keypoint_data)

        # Button to run remove_anything.py
        if st.button("Run remove_anything.py"):
            if canvas.json_data["objects"]:
                # Extract the coordinates of the keypoint from the canvas data
                keypoint_data = canvas.json_data["objects"][0]
                keypoint_x = keypoint_data["top"]
                keypoint_y = keypoint_data["left"]
                print(keypoint_x,keypoint_y)
                # Run remove_anything.py with keypoint as command line arguments
                os.system("python "+ "remove_anything.py "+ f' --input_img /home/sam/Pictures/monalisa.jpg  --coords_type key_in  --point_coords {str(keypoint_x)} {str(keypoint_y)}  --point_labels 1  --dilate_kernel_size 15  --output_dir ./results  --sam_model_type "vit_h" --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth --lama_config ./lama/configs/prediction/default.yaml --lama_ckpt ./pretrained_models/big-lama')
                st.write("remove_anything.py executed successfully.")
            else:
                st.write("Please select a keypoint by clicking on the image.")

if __name__ == "__main__":
    main()