# Source到csv的路径
SOURCE_REVIEW_JSON=data/google_local-Alabama/review-Alabama_10w.json
SOURCE_META_JSON=data/google_local-Alabama/meta-Alabama.json
TEST_RATIO=0.1

# 标注需要的csv路径
ORI_CSV_PATH=data/google_local-Alabama/train.csv
GT_CSV=data/google_local-Alabama/train_gt.csv
BUSINESS_COL=store_info
COMMENT_COL=text_info


# 联合meta和review数据
python data_processing/join_google_local_data.py --meta-file $SOURCE_META_JSON --review-file $SOURCE_REVIEW_JSON --output-file $ORI_CSV_PATH
# 提取store和text信息，过滤掉空的text or description
python data_processing/extract_store_text.py --input-file $ORI_CSV_PATH --output-file $ORI_CSV_PATH
# 划分train和test
python data_processing/split_train_test.py --input-file $ORI_CSV_PATH --output-file $ORI_CSV_PATH --test-ratio $TEST_RATIO
# GPT-4o 请求失败的filter out
python data_processing/clean_train_data.py --input-file $ORI_CSV_PATH --output-file $ORI_CSV_PATH --error-pattern "ERROR: Error calling OpenAI API: Err"
# Output the data distribution, column is "ai_classification"
python data_processing/data_analysis.py $GT_CSV
# GPT-4o标注
python review_classifier.py --csv-file $ORI_CSV_PATH --output-file data/store_text_extracted_english_classified.csv --store-column $BUSINESS_COL --comment-column $COMMENT_COL --workers 32 --output-file $GT_CSV
