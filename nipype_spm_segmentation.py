# code created by Sudhakar on June 2020
# multi-modal spm segmentation 

import os
import nipype_all_functions as naf

tpm_image = '/usr/users/nmri/tools/spm/spm12_7487/tpm/TPM.nii'

def get_file_name_and_extension(infile):
    '''
    Parameters
    ----------
    infile : str
        filename.

    Returns
    -------
    TYPE: str
        main name of the filename.
    TYPE: str
        extension of the filename.
    '''
    main_1, ext_1 = os.path.splitext(infile)
    if ext_1 == '.gz':
        main_2, ext_2 = os.path.splitext(main_1)
        return main_2, ext_2+ext_1
    else:
        return main_1, ext_1

def do_spm_segmentation(data_path, subject, image_type, multi_channel):
    '''
    Parameters
    ----------
    data_path : str
        full path to the data_path.
    subject : str
        subject ID.
    image_type : str
        it is either 'anat' or 'align.

    Returns
    -------
    gray matter, white matter and CSF probability maps in native space fot the subject 

    '''
    raw_path = os.path.join(data_path, subject, 'anat') # raw images
    align_path = os.path.join(data_path, subject, 'align') # rigidly aligned images
    mats_path = os.path.join(data_path, subject, 'mat') # transformation matrix files
    
    T1_flag = False
    T2_flag = False
    FLAIR_flag =  False
    
    if image_type == 'anat':
        if os.path.getsize(raw_path) > 0:
            raw_images = os.listdir(raw_path)
            for raw_image in raw_images:
                if raw_image.endswith('reoriented.nii.gz') and 'hrT1' in raw_image:
                    T1_image = raw_image
                    T1_flag = True
                elif raw_image.endswith('reoriented.nii.gz') and 'hrT2' in raw_image:
                    T2_image = raw_image
                    T2_flag = True
                elif raw_image.endswith('reoriented.nii.gz') and 'hrFLAIR' in raw_image:
                    FLAIR_image = raw_image
                    FLAIR_flag = True
            if T1_flag:
                print(f'raw T1-image is present for the subject {subject}')
                if T2_flag:
                    print(f'both T1 and T2 raw images are present for {subject}')
                    for mat_file in os.listdir(mats_path):
                        if mat_file.endswith('t2-t1.mat'):
                            main_name, ext = get_file_name_and_extension(T2_image)
                            outfile = main_name+'.alignedToT1'+ext
                            naf.doApplyXFM(os.path.join(raw_path, T2_image), os.path.join(mats_path, mat_file), os.path.join(raw_path, T1_image), os.path.join(raw_path, outfile), 'spline', 'T2')
                            if multi_channel:
                                naf.do_spm_new_segment_multi_channel(tpm_image, os.path.join(raw_path, T1_image), os.path.join(raw_path, T2_image))
                            else:
                                naf.do_spm_new_segment(tpm_image, os.path.join(raw_path, T1_image))
                elif FLAIR_flag:
                    print(f'both T1 and FLAIR raw images are present for {subject}')
                    for mat_file in os.listdir(mats_path):
                        if mat_file.endswith('flair-t1.mat'):
                            main_name, ext = get_file_name_and_extension(FLAIR_image)
                            outfile = main_name+'.alignedToT1'+ext
                            naf.doApplyXFM(os.path.join(raw_path, FLAIR_image), os.path.join(mats_path, mat_file), os.path.join(raw_path, T1_image), os.path.join(raw_path, outfile), 'spline', 'FLAIR')
                            if multi_channel:
                                naf.do_spm_new_segment_multi_channel(tpm_image, os.path.join(raw_path, T1_image), os.path.join(raw_path, FLAIR_image))
                            else:
                                naf.do_spm_new_segment(tpm_image, os.path.join(raw_path, T1_image))
                elif T2_flag and FLAIR_flag:
                    print(f'all T1, T2 and FLAIR raw image are present for {subject}, using FLAIR along with T1')
                    for mat_file in os.listdir(mats_path):
                        if mat_file.endswith('flair-t1.mat'):
                            main_name, ext = get_file_name_and_extension(FLAIR_image)
                            outfile = main_name+'.alignedToT1'+ext
                            naf.doApplyXFM(os.path.join(raw_path, FLAIR_image), os.path.join(mats_path, mat_file), os.path.join(raw_path, T1_image), os.path.join(raw_path, outfile), 'spline', 'FLAIR')
                            if multi_channel:
                                naf.do_spm_new_segment_multi_channel(tpm_image, os.path.join(raw_path, T1_image), os.path.join(raw_path, FLAIR_image))
                            else:
                                naf.do_spm_new_segment(tpm_image, os.path.join(raw_path, T1_image))
                else:
                    print(f'only T1 image is present for the subject {subject}')
                    naf.do_spm_new_segment(tpm_image, os.path.join(raw_path, T1_image))
        else:
            print(f'{raw_path} is empty moving on to next subject')      
    elif image_type == 'align':
        if os.path.getsize(align_path) > 0:
            align_images = os.listdir(align_path)
            for align_image in align_images:
                if align_image.endswith('reoriented.align.nii.gz') and 'hrT1' in align_image:
                    T1_image = align_image
                    T1_flag = True
                elif align_image.endswith('reoriented.align.nii.gz') and 'hrT2' in align_image:
                    T2_image = align_image
                    T2_flag = True
                elif align_image.endswith('reoriented.align.nii.gz') and 'hrFLAIR' in align_image:
                    FLAIR_image = align_image
                    FLAIR_flag = True
            if T1_flag:
                print(f'T1-image is present for the subject {subject}')
                if T2_flag:
                    print(f'both T1 and T2 aligned images are present for {subject}')
                    if multi_channel:
                        naf.do_spm_new_segment_multi_channel(tpm_image, os.path.join(align_path, T1_image), os.path.join(align_path, T2_image))
                    else:
                        naf.do_spm_new_segment(tpm_image, os.path.join(align_path, T1_image))
                elif FLAIR_flag:
                    print(f'both T1 and FLAIR aligned images are present for {subject}')
                    if multi_channel:
                        naf.do_spm_new_segment_multi_channel(tpm_image, os.path.join(align_path, T1_image), os.path.join(align_path, FLAIR_image))
                    else:
                        naf.do_spm_new_segment(tpm_image, os.path.join(align_path, T1_image))
                elif T2_flag and FLAIR_flag:
                    print(f'all T1, T2 and FLAIR raw image are present for {subject}, using FLAIR along with T1')
                    if multi_channel:
                        naf.do_spm_new_segment_multi_channel(tpm_image, os.path.join(align_path, T1_image), os.path.join(align_path, FLAIR_image))
                    else:
                        naf.do_spm_new_segment(tpm_image, os.path.join(align_path, T1_image))
                else:
                    print(f'only aligned T1 image is present for the subject {subject}')
                    naf.do_spm_new_segment(tpm_image, os.path.join(align_path, T1_image))
        else:
            print(f'{align_path} is empty moving on to next subject') 
    else:
        print(f'wrong {image_type} is considered, it should be either anat or align')
                