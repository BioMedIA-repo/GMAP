import torch

def reverse_mode(mode):
    if mode == "CT":
        return 'MR'
    elif mode == "MR":
        return 'CT'
    elif mode == "ABCT":
        return 'ABMR'
    else:
        return 'ABCT'



def get_mid_image(source_images, target_images):
    mid_images = 0.5 * source_images.clone() + 0.5 * target_images.clone()
    mid_images = mid_images.to(torch.float32).cuda()
    return mid_images



@torch.no_grad()
def ema_update(student_model, teacher_model, alpha=0.999):
    student_state_dict = student_model.state_dict()

    for key, teacher_param in teacher_model.state_dict().items():
        student_param = student_state_dict[key]

        if teacher_param.dtype.is_floating_point:
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
        else:
            teacher_param.data.copy_(student_param.data)
