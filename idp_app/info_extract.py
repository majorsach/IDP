import subprocess
import os
from django.conf import settings
def run_inference(infer_img, use_gpu):
    paddle_ocr_dir = os.path.join(settings.BASE_DIR, 'idp_app', 'PaddleOCR-release-2.7')

    original_dir = os.getcwd()
    # Change the current working directory to "PaddleOCR-release-2.6"
    os.chdir(paddle_ocr_dir)
    # save_res_path= os.path.join(settings.BASE_DIR, 'idp_app', 'static','out_images')
    print('processing...')
    print("INFERIMAGEE::::",infer_img)
    # command = [
    #     "python",
    #     "./tools/infer_kie_token_ser_re.py",
    #     "-c",
    #     "configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml",
    #     "-o",
    #     f"Architecture.Backbone.checkpoints=./pretrained_model/re_vi_layoutxlm_xfund_pretrained/best_accuracy",
    #     f"Global.infer_img={infer_img}",
    #     f"Global.use_gpu={str(use_gpu)}",
    #     "-c_ser",
    #     "configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml",
    #     "-o_ser",
    #     f"Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy"
    # ]

    #just SER

    command = [
        "python",
        "./tools/infer_kie_token_ser.py",
        "-c",
        "configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml",
        "-o",
        f"Architecture.Backbone.checkpoints= ./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy",
        f"Global.infer_img={infer_img}",
        f"Global.use_gpu={str(use_gpu)}",
         ]


    subprocess.run(command)
    os.chdir(original_dir)

def main_inference(file_path):
    # Call the function with the desired arguments
    run_inference(file_path, use_gpu=False)
    # os.chdir("PaddleOCR-release-2.6")
