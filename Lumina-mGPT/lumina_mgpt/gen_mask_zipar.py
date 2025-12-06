import torch

def cal_mask_zipar(h_seq_len,w_seq_len,text_len,window_sz=8):
    # h_seq_len,w_seq_len:不包含特殊token的长度
    # text_len:text部分所有token的长度
    img_seq_len=h_seq_len*(w_seq_len+1)
    total_seq_len=img_seq_len+4+text_len
    causal_mask = torch.tril(torch.ones(img_seq_len, img_seq_len, dtype=torch.bool))
    line_len=w_seq_len+1#每一行包含的token数
    k_max=int(line_len//window_sz)
    row=torch.arange(img_seq_len).view(-1,1)
    col=torch.arange(img_seq_len).view(1,-1)
    row_idx=torch.arange(h_seq_len).repeat_interleave(line_len).view(-1,1)
    col_idx=torch.arange(h_seq_len).repeat_interleave(line_len).view(1,-1)
    for k in range(k_max,0,-1):
        causal_mask[(row-col)<=k*(line_len-window_sz)]=False
        causal_mask[(row_idx-col_idx)<k]=True
        causal_mask[:,(((row%line_len)==0) & (row!=0)).view(-1)]=True#eol对所有token都可见
        # mask_image = causal_mask.to(dtype=torch.uint8) * 255
        # Image.fromarray(mask_image.numpy(), mode="L").save("causal_mask_%d.png"%k)

    causal_mask_=torch.ones(total_seq_len-1, total_seq_len-1, dtype=torch.bool)
    causal_mask_[text_len+3:,text_len+3:]=causal_mask
    causal_mask_=torch.tril(causal_mask_)
    # mask_image = causal_mask_.to(dtype=torch.uint8) * 255
    # Image.fromarray(mask_image.numpy(), mode="L").save("causal_mask_.png")
    return causal_mask_

if __name__=='__main__':
    cal_mask_zipar(h_seq_len=16,w_seq_len=24,text_len=30,window_sz=4)