for NUM in $(seq 1 3); do
  echo "Opening Colmap for Split $NUM"
  DATABASE_PATH="data/data_for_gaussian_splatting/reference_bag/colmap/database_split_${NUM}.db"
  IMAGE_PATH="data/data_for_gaussian_splatting/reference_bag/images_gs_split_${NUM}_1_of_3"
  OUTPUT_PATH="data/data_for_gaussian_splatting/reference_bag/colmap/split_${NUM}/sparse/0"
  #colmap gui --database_path="$DATABASE_PATH" --image_path="$IMAGE_PATH"

  LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 colmap feature_extractor \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_PATH" \
  --ImageReader.camera_model OPENCV \
  --ImageReader.camera_params "785.34926249, 784.07587341, 406.50794975, 249.45341029, -0.42020115, 0.64296938, -0.00531934, -0.00215015"

  LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 colmap sequential_matcher \
  --database_path "$DATABASE_PATH" \
  --SiftMatching.guided_matching 1 \
  --SequentialMatching.overlap 10

  mkdir "$OUTPUT_PATH"

  LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0 colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH" \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.ba_refine_principal_point 0

done