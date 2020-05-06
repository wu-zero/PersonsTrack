#PersonsTrack
#目录

# 简介
利用计算机视觉的技术（目标检测、人体关键点检测、人脸识别、目标追踪）实现室内人员的大致定位以及身份匹配。用于餐厅可以实现针对人群的菜品推荐以及无感人脸支付。  

其中：  
* 目标检测采用YOLOV3模型  
* 人体关键点检测采用openpose模型  
* 人脸识别采用商用的人脸识别服务器  
* 目标追踪采用DeepSort模型  

关键技术：
* 对10路视频实时采集视频流并定时分发给人体追踪进程和人脸检测进程。
* 设计人体框和人脸框匹配规则，计算匹配置信度，利用Kuhn-Munkres算法为人员追踪定位结果匹配身份信息
* 基于人体信息结合多项约束设计人员定位算法


结果展示：
<div align=center><img width="700"  src="https://github.com/wu-zero/PersonsTrack/raw/master/docs/imgs/result.png"/></div>  

# 主要依赖
* pip安装  
    ```
    NumPy  
    sklean  
    OpenCV  
    Pillow  
    PyMySQL  
    requests  
    PyRSMQ  
    Cython  
    multiprocessing-logging  
    ```
* conda安装  
    ```
    tensorflow-gpu 1.15
    ```

# 效果过程
* 人脸、人体匹配效果
  * 方法一
    <div align=center><img width="700"  src="https://github.com/wu-zero/PersonsTrack/raw/master/docs/imgs/%E5%8C%B9%E9%85%8D%E6%96%B9%E6%B3%951.jpg"/></div>  
  * 方法二
    <div align=center><img width="700"  src="https://github.com/wu-zero/PersonsTrack/raw/master/docs/imgs/%E5%8C%B9%E9%85%8D%E6%96%B9%E6%B3%952.jpg"/></div>
* 定位效果展示
  * 图一
    <div align=center><img width="700"  src="https://github.com/wu-zero/PersonsTrack/raw/master/docs/imgs/%E5%AE%9A%E4%BD%8D%E7%BB%93%E6%9E%9C%E7%A4%BA%E6%84%8F%E5%9B%BE1.png"/></div>  
  * 图二
    <div align=center><img width="700"  src="https://github.com/wu-zero/PersonsTrack/raw/master/docs/imgs/%E5%AE%9A%E4%BD%8D%E7%BB%93%E6%9E%9C%E7%A4%BA%E6%84%8F%E5%9B%BE2.png"/></div>  

