cp_h, cp_w = 14, 56     # origin is (14, 56) change to (28, 28)
roi_h, roi_w = 7, 28

nms_ct = True
max_ct_overlap = 0.7    # origin is 0.7, 0.6 better
ct_score = 0.03         # 0.03 is better, origin is 0.05

cp_hm_nms = False # why
max_cp_det = 50
max_cp_overlap = 0.1
cp_score = 0.25         # origin is 0.25, 0.3 better

segm_or_bbox = 'segm'