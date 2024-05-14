import albumentations as A

def test_transforms(crop_size, mean=0.0, std=1.0, augment=True, norm=True):
    transforms = []

    transforms.append(A.Resize(crop_size[0], crop_size[1], interpolation=1))
    if norm:
        transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))

    return A.Compose(transforms)

def train_transforms(crop_size, mean=0.0, std=1.0, augment=True, norm=True):
    transforms = []

    transforms.append(A.Resize(crop_size[0], crop_size[1], interpolation=1))
    if augment:
        # transforms.append(A.Rotate(limit=360, interpolation=1, border_mode=4, always_apply=True))
        transforms.append(A.HorizontalFlip(p=0.5))
        #transforms.append(A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1))
        #transforms.append(A.RandomCrop(crop_size, crop_size, always_apply=True))
        #transforms.append(A.Resize(crop_size, crop_size, interpolation=1))
        transforms.append(A.MotionBlur())
        transforms.append(A.Rotate(limit=10))
        transforms.append(A.RandomBrightnessContrast(0.75, 1.25, p=1.0))
    if norm:
        transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))

    return A.Compose(transforms)

def embedded_transforms(crop_size, mean=0.0, std=1.0):
    transforms = []


    transforms.append(A.Resize(crop_size[0], crop_size[1], interpolation=1))
    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
    transforms.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
    transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))
    transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1))





    return A.Compose(transforms)

def ssl_transforms(crop_size, mean=0.0, std=1.0,norm=True):
    transforms = []


    transforms.append(A.Resize(crop_size[0], crop_size[1], interpolation=1))
    transforms.append(A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5))
    transforms.append(A.ElasticTransform(alpha=80, sigma=120 * 0.05, alpha_affine=80 * 0.03, p=0.5))
    transforms.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5))
    transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))
    transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1))
    if norm:
        transforms.append(A.Normalize(mean=mean, std=std, always_apply=True))


    return A.Compose(transforms)