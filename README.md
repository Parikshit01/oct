# oct

The method uses Dense Residual Swin Transformer for Image Denoising followed by META Segment Anything SAM Model for ROI based segmentation for finding the disease detection part on OCT Eye images.

@inproceedings{li2023ntire_dn50,
  title={NTIRE 2023 Challenge on Image Denoising: Methods and Results},
  author={Li, Yawei and Zhang, Yulun and Van Gool, Luc and Timofte, Radu and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2023}
} 


@misc{kirillov2023segment,
      title={Segment Anything}, 
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

conda create --name oct
Steps to follow:-
pip install -r requirements.txt


DESKTOP VERSION (Streamlit App)
run 'streamlit run basic.py'

download the model files from below links and keep it in the root directory
https://drive.google.com/file/d/1NUjLkEI6d0e0KuFC00PNHX_aFYJWSaJf/view?usp=sharing
https://drive.google.com/file/d/1nkPqglbnNo7gQ6fGmS6vlibQOybgnqhZ/view?usp=sharing
https://drive.google.com/file/d/1Fsw2G-HFFMgFa-Kzi-MmhrjLSjcvgydm/view?usp=sharing
https://drive.google.com/file/d/1wiNby1oIEPFohgeOdrtg4ZHy5f_2mhei/view?usp=sharing

or using gdown library
like 
 
 pip3 install gdown
 
 gdown 1NUjLkEI6d0e0KuFC00PNHX_aFYJWSaJf 
 
 gdown 1nkPqglbnNo7gQ6fGmS6vlibQOybgnqhZ
 
 gdown 1Fsw2G-HFFMgFa-Kzi-MmhrjLSjcvgydm
 
 gdown 1wiNby1oIEPFohgeOdrtg4ZHy5f_2mhei

