train_lr_tensors = load_images(r'NewDataSet\Train\LR256')
train_hr_tensors = load_images(r'NewDataSet\Train\HR256')
valid_lr_tensors = load_images(r'div2k\DIV2K_valid_LR_bicubic_X2\DIV2K_valid_LR_bicubic\X2')
valid_hr_tensors = load_images(r'div2k\DIV2K_valid_HR\DIV2K_valid_HR')
test_lr_tensors = load_images(r'div2k\DIV2K_test_LR_bicubic_X2\DIV2K_test_LR_bicubic\X2\Set5')
test_hr_tensors = load_images(r'div2k\DIV2K_test_HR\DIV2K_test_HR\Set5')

train_dataset = DIV2KDataset(train_lr_tensors[:33240], train_hr_tensors[:33240])
valid_dataset = DIV2KDataset(valid_lr_tensors[:100], valid_hr_tensors[:100])
test_dataset = DIV2KDataset(test_lr_tensors[:10], test_hr_tensors[:10])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
