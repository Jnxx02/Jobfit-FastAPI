# Jobfit API
API untuk aplikasi JobFit. Dibuat dengan FastAPI.

## Note
Endpointnya bisa semua diakses dengan link beriku `https://jobfit-api-0ea5f7774884.herokuapp.com/` kecuali `/top_matches`, aksesnya harus menggunakan postman karena methodnya post.

## Endpoint
`/` : Halaman utama

![image](https://github.com/user-attachments/assets/81ddaa78-b008-4fbd-af9d-49c5ca332903)

`/match_job` : tampilkan kecocokan user dalam bentuk pie chart (fungsinya mirip dengan top_matches, cuma beda di penyajian data).

![image](https://github.com/user-attachments/assets/0d3caa12-3c66-4837-ad1e-3deb5fb21452)

`/companies_jobs` : tampilkan company dan job yang ditawarkan.

![image](https://github.com/user-attachments/assets/62a56241-4618-49d4-b7fa-66969b47a18b)

`/top_matches` : tampilkan 5 matching job terbaik berdasarkan skills dan experience (sama seperti fungsi yang kemarin).

![image](https://github.com/user-attachments/assets/c1450780-d1b8-4025-97fa-e4077859b56a)
