# ANIMAL CLASSIFICATION

Animal Classification with Baseline CNN Architecture, ResNet50 Transfer Learning, Few-shot Learning with Prototypical Networks, One-shot Learning with Siamese Networks and Contrastive Learning with SimCLR

## BÁO CÁO DỰ ÁN 

[Deep Learning.pdf](https://github.com/user-attachments/files/24382267/Deep.Learning.pdf)

## GIỚI THIỆU

**Một nghiên cứu toàn diện về nhận diện phân loại ảnh động vật sử dụng nhiều phương pháp học sâu khác nhau.** Dự án này không chỉ dừng lại ở việc đạt độ chính xác cao trên các lớp đã biết, mà còn khám phá khả năng nhận diện các lớp chưa từng thấy (unseen classes) thông qua các kỹ thuật tiên tiến như Few-Shot Learning, One-Shot Learning, và Contrastive Learning.

## TỔNG QUAN DỰ ÁN

### 1. Supervised Learning (Baseline CNN Architecture & Transfer Learning ResNet50)

#### Mục tiêu: Xây dựng một baseline vững chắc và tận dụng tri thức từ ImageNet để phân loại chính xác các lớp đã biết (Seen Classes).

##### Tư duy triển khai:

- Bắt đầu từ một kiến trúc CNN tùy chỉnh để kiểm chứng dữ liệu, sau đó chuyển sang Transfer Learning với ResNet50.

- Sử dụng Global Average Pooling (GAP) để giảm chiều dữ liệu thay vì Flatten, giúp giảm tham số và hạn chế Overfitting.

##### Luồng xử lý:

1. Input: Ảnh đầu vào được Resize (256x256) và CenterCrop (224x224).

2. Feature Extraction:

    - Custom CNN: Đi qua 4 khối Conv -> BatchNorm -> ReLU -> MaxPool.
    
    - ResNet50: Đóng băng (Freeze) toàn bộ các layer đầu, chỉ mở khóa huấn luyện (Unfreeze) layer4 và fc.

3. Classification Head: Global Average Pool -> Dropout (0.3) -> Linear Layer (512 nodes) -> Output Classes.

##### Điểm nhấn triển khai:

- Imbalance Handling: Sử dụng compute_class_weight với chiến lược 'balanced' để gán trọng số lớn hơn cho các lớp ít mẫu vào hàm Loss.

- Cross-Validation: Triển khai K-Fold (k=4) để đảm bảo độ tin cậy của mô hình trên tập dữ liệu nhỏ.

### 2. Meta-Learning (Few-Shot Learning)

#### Mục tiêu: Học cách học (Learning to Learn). Cho phép mô hình nhận diện lớp mới (ví dụ: Pig) chỉ sau khi nhìn thấy vài ví dụ (5-shot) mà không cần train lại toàn bộ mạng.

##### Tư duy triển khai:

- Sử dụng Prototypical Networks (ProtoNet).

- Tư duy theo mô hình "Tập" (Episodic Training): Thay vì Epoch thông thường, quá trình train được chia thành các Episode (N-way K-shot).

##### Luồng xử lý:

- Episode Construction: Với mỗi episode, chọn ngẫu nhiên $N$ lớp (5-way). Mỗi lớp lấy $K$ ảnh làm Support Set và $Q$ ảnh làm Query Set.

- Prototype Calculation: Tính trung bình vector đặc trưng của Support Set để tạo ra "Prototype" ($c_k$) cho mỗi lớp:

  $$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

- Classification: Tính khoảng cách từ ảnh Query đến các Prototypes và dùng Softmax đê phân loại.

##### Điểm nhấn triển khai:

- Optimization: Sử dụng GradScaler và autocast để huấn luyện với độ chính xác hỗn hợp (Mixed Precision), giúp tăng tốc độ và giảm bộ nhớ GPU.

- Scheduler: Sử dụng CosineAnnealingLR để điều chỉnh learning rate mượt mà, giúp hội tụ tốt hơn vào cuối quá trình train.

<img width="280" height="210" alt="seen_confusion_matrix" src="https://github.com/user-attachments/assets/97ece675-34aa-4d59-bcc1-605f16174b26" />
<img width="280" height="210" alt="pig_confusion_matrix" src="https://github.com/user-attachments/assets/1b0307a1-e8ac-494c-96f7-cbdc41257588" />
<img width="280" height="210" alt="tsne_FewshotLearning" src="https://github.com/user-attachments/assets/0e18b6ed-b790-4e66-9588-c6be4ce87eda" />


### 3. Metric Learning (One-Shot Learning)

#### Mục tiêu: Nhận diện bất thường (Anomaly/OOD Detection). Hệ thống có thể từ chối nhận diện hoặc gán nhãn "Unknown" cho các lớp lạ (ví dụ: Butterfly) dựa trên khoảng cách.

##### Tư duy triển khai: Thay vì học "đây là con chó", mô hình học "hai ảnh này có giống nhau không?".

- Sử dụng kiến trúc Siamese Network với hai nhánh chia sẻ trọng số (Shared Weights).

##### Luồng xử lý:

1. Pair Generation: Dataset tạo ra các cặp ảnh:

    - Positive Pair: Hai ảnh cùng class (Label = 1).
 
    - Negative Pair: Hai ảnh khác class (Label = 0).

2. Siamese Backbone: Cả hai ảnh đi qua cùng một ResNet backbone để ra hai vector đặc trưng $v_1, v_2$.

3. Distance Metric: Tính khoảng cách Euclidean: $d = ||v_1 - v_2||_2$.

4. Contrastive Loss: Tối ưu hóa hàm loss để $d \to 0$ nếu cùng lớp, và $d > margin$ nếu khác lớp:

      $$L = y \cdot d^2 + (1-y) \cdot \max(0, margin - d)^2$$.

#### Điểm nhấn triển khai: 

Thresholding: Phân tích ROC Curve để tìm ra ngưỡng tối ưu (0.3711) giúp phân tách lớp Butterfly (Unseen) ra khỏi các lớp Seen với độ chính xác ~93.84%.

<img width="1000" height="750" alt="normalized_distance_distribution" src="https://github.com/user-attachments/assets/9f42c896-37ed-40a1-b967-80e5881ccefc" />


### 4. Self-Supervised Learning (Contrastive - SimCLR)
#### Mục tiêu: Học biểu diễn đặc trưng (Feature Representation) mạnh mẽ mà không cần nhãn. Đây là bước chuẩn bị tuyệt vời khi dữ liệu gán nhãn khan hiếm.

##### Tư duy triển khai:

Sử dụng framework SimCLR. Ý tưởng là một ảnh khi bị biến đổi (augment) mạnh vẫn phải có đặc trưng gần giống với bản gốc của nó (Positive Pair) và xa các ảnh khác (Negative Pair).

##### Luồng xử lý:

1. Heavy Augmentation: Một ảnh gốc được nhân bản thành 2 phiên bản qua các phép biến đổi mạnh: RandomResizedCrop, ColorJitter, GaussianBlur.

2. Encoder & Projection: Đưa qua ResNet50 (Encoder) và một Projection Head (MLP: 2048 -> 512 -> 256).

3. NT-Xent Loss: Tối ưu hóa hàm loss dựa trên nhiệt độ (Temperature 0.1) để kéo các cặp ảnh giống nhau lại gần trong không gian cầu (hypersphere).

##### Điểm nhấn triển khai:

Evaluation: Kiểm chứng bằng Linear Evaluation Protocol (train một Linear Classifier trên feature đã đóng băng) và trực quan hóa bằng t-SNE để chứng minh các cụm dữ liệu phân tách rõ ràng.

<img width="400" height="300" alt="confusion_matrix_simclr_final" src="https://github.com/user-attachments/assets/f51bb0b1-138f-4530-9f8f-e77a6a960fbe" />
<img width="400" height="300" alt="tsne_simclr_final" src="https://github.com/user-attachments/assets/937bf751-3f8e-4097-98d4-a7355beb3b5e" />
