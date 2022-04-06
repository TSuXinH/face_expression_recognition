本次作业的目录结构以及相应的注释如下：

##############################
│  readme.txt
│  report.pdf		实验报告
│  
└─code_etc
    │  API_UI_class.py	API 和 UI 之间的接口，运行此文件打开 UI（选做）
    │  a_net.pkl		VGG10 模型: test_acc： .725
    │  b_net.pkl		resnet25 模型: test_acc： .704
    │  c_net.pkl		VGG10 模型: test_acc： .733
    │  data_process.py		预处理数据
    │  execution.py			线下测试代码文件，和提供的 notebook 文件结构类似，可以不运行
    │  image_form.py		image ui 转 py 文件
    │  img_video_API.py		处理图像和视频的原始 API （选做）
    │  model.py			网络模型和训练测试等接口
    │  settings_form.py		settings ui 转 py 文件
    │  start_form.py			start ui 转 py 文件
    │  task1.ipynb			必做一 notebook 文件
    │  task2.ipynb			必做二 notebook 文件
    │  utility.py			各种辅助函数
    │  video_form.py			video ui 转 py 文件
    │  
    ├─result			视频输出结果
    │      output_video.mp4
    │       
    │  
    ├─test_image_video		用于测试的图片和视频
    │  ├─image
    │  │      baodai.jpg
    │  │      girl.jpg
    │  │      test_lenna.jpg
    │  │      you.jpg
    │  │      zhang.jpg
    │  │      
    │  └─video
    │          v5.mp4
    │          
    ├─ui_designer			ui 文件
    │      image_form.ui
    │      settings_form.ui
    │      start_form.ui
    │      video_form.ui
    │      
    └─ui_relatives			ui 使用到的图片
             bg.jpg
             icon.ico
##############################

运行时直接按照此目录运行即可，无需进行修改。
模型的位置如文件树所示，模型下载链接为 https://cloud.tsinghua.edu.cn/d/61462b722f664fc9aa1f/
ui 的注意事项详见 report.pdf.
