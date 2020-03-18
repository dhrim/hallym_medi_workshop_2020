# 딥러닝 파악

- 라이브러리와 linux : [딥러닝을 위한 라이브러리와 linux](https://docs.google.com/presentation/d/1d58RGoqrUyVa8NpWPLsyqcZuEXXaurosTRwjBgyYvPE/edit?usp=sharing)

- 딥러닝 개요
    - 딥러닝 개념 : [deep_learning_intro.pptx](material/deep_learning_intro.pptx)
    - 알파고 이해하기 : [understanding_alphago.pptx](material/understanding_alphago.pptx)
- Keras
    - DNN in Keras : [dnn_in_keras.ipynb](material/dnn_in_keras.ipynb)
    - Keras 요약 [keras_in_short.md](material/keras_in_short.md)
- DNN as classifier
    - 분류기 : [dnn_as_a_classifier.ipynb](material/dnn_as_a_classifier.ipynb)
    - IRIS
        - [dnn_iris_and_optimizer.ipynb](material/dnn_iris_and_optimizer.ipynb)
        - [dnn_iris_with_category_index.ipynb](material/dnn_iris_with_category_index.ipynb)
    - MNIST 영상데이터 : [dnn_mnist.ipynb](material/dnn_mnist.ipynb)
- CNN
    - MNIST : [cnn_mnist.ipynb](material/cnn_mnist.ipynb)
    - CIFAR10 : [cnn_cifar10.ipynb](material/cnn_cifar10.ipynb)
    - IRIS : [iris_cnn.ipynb](material/iris_cnn.ipynb)
- VGG - CIFAR10, ImageNet, custom data : [VGG16_classification_and_cumtom_data_training.ipynb](material/VGG16_classification_and_cumtom_data_training.ipynb)
- U-Net Segmentation - Lung data : [unet_segementation.ipynb](material/unet_segementation.ipynb)
- Object Detection
    - CoCo, custom data : [object_detection.md](material/object_detection.md)
- Text : [text_classification.ipynb](material/text_classification.ipynb)

- 손실함수의 이해 : [understanding_loss_function.ipynb](material/understanding_loss_function.ipynb)

<br>

# 딥러닝 논문 파악

- [딥러닝 논문 리뷰](https://docs.google.com/presentation/d/1SZ-m4XVepS94jzXDL8VFMN2dh9s6jaN5fVsNhQ1qwEU/edit?usp=sharing)


<br>

# 기타

- [디노이징 AutoEncoder](material/denoising_autoencoder.ipynb)
- [흥미로운 딥러닝 결과](material/some_interesting_deep_learning.pptx)
- [yolo를 사용한 실시간 불량품 탐지 사례](material/yolo_in_field.mp4)
- [GAN을 사용한 생산설비 이상 탐지](material/anomaly_detection_using_gan.pptx)
- [이상탐지 동영상](material/drillai_anomaly_detect.mp4)


<br>

# 딥러닝 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
linux
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        op
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
        to numpy
    matplot
        line graph
        scatter graph
        show image

Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tunning
            overfitting, underfitting
            regularization, dropout, batch normalization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            varnilla, VGG16
        RNN
        GAN
    task
        Classification
        Object Detection
        Segmentation
        Anomaly Detection
        Generation
        target : text/image

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate
        regularier
        dropout
        batch normalization
    save/load
    compile
        optimizer
        loss
        metric
```

<br>