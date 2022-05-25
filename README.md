# CS331.M22.KHCL

# THÀNH VIÊN TRONG NHÓM:
| STT    | MSSV          | Họ và Tên              |Chức Vụ     | Github                                                  | Email                   |
| ------ |:-------------:| ----------------------:|-----------:|--------------------------------------------------------:|-------------------------:
| 1      | 19521797     | Phạm Minh Long  |Thành Viên  |[LongPML](https://github.com/LongPML)            |19521797@gm.uit.edu.vn   |
| 2      | 19521943      | Phan Nguyễn Thành Nhân        |Thành viên  |[thanhnhan311201](https://github.com/thanhnhan311201)                |19521943@gm.uit.edu.vn   |
| 3      | 19521634      | Tạ Huỳnh Đức Huy        |Thành viên  |[huy](https://github.com/)                |19521634@gm.uit.edu.vn   |
| 4      | 19521764      | Nguyễn Trần Phước Lộc        |Thành viên  |[ntploc0910](https://github.com/ntploc0910)                |19521764@gm.uit.edu.vn   |

# GIỚI THIỆU MÔN HỌC
* **Tên môn học:** Thị Giác Máy Tính Nâng Cao
* **Mã môn học:** CS331
* **Mã lớp:** CS331.M22.KHCL
* **Năm học:** HK2 (2021 - 2022)
* **Giảng viên**: Mai Tiến Dũng

# Hướng dẫn chạy

## Xây dựng database

Chạy đoạn code sau đây để xây dựng database.

```bash
python3 demo.py models/83_epochs/ --option index --dataset_path datasets/
```

## Chạy demo với ảnh

Chạy đoạn code sau đây để chạy demo với ảnh.

```bash
python3 demo.py models/83_epochs/ --dataset_path datasets/ --option demo_via_img --image_file your_image_file
```

## Chạy demo với camera

Chạy đoạn code sau đây để chạy demo với camera.

```bash
python3 demo.py models/83_epochs/ --dataset_path datasets/ --option demo_via_cam
```