# 对张量元素进行二进制分解和打包，以及重构
import math
import numpy as np
import torch

# 此函数有两个输入参数，一个整数 x 和一个整数比特数 n_bits。它将整数 x 分解为其位表示形式，并返回一个与比特向量大小相同的
布尔张量。表示 x 的位数由 n_bits 参数指定。
def bitdecomp(x, n_bits):
    mods = []
    for _ in range(n_bits):
        mods.append(x % 2)
        x = x / 2

    bitrep = torch.stack(mods, dim=-1).byte()

    return bitrep

# 此函数有三个输入参数，一个布尔张量 x，大小为 b_source，即原数的比特数；一个整数 b_source，指定原数的比特数；一个整数 b_target。它使用输入的布尔张量 x 重构原始数字，并将重构后的数字以短张量形式返回，大小为一。
def bitrecon(x, b_source, b_target):
    bitrep = x.view(b_source * len(x) // b_target, b_target)
    exp = torch.ShortTensor([2**e for e in range(b_target)]).view(1, b_target)
    recon = (bitrep.short() * exp).sum(dim=1).view(-1)

    return recon

# 此函数有两个输入参数，一个张量 n 和一个整数比特数 n_bits。它将输入的张量 n 扁平化，添加填充零以使其长度可被 8 整除，然后使用函数 bitdecomp() 将其转换为布尔张量。最后，使用 bitrecon() 将 n 重构为一个 8 位无符号整型张量。函数返回结果为一个 8 位张量。
def numpack(n, n_bits):
    flat = n.view(-1)
    if len(flat) % 8 > 0:
        flat = torch.cat((flat, flat.new_zeros(8 - len(flat) % 8)))

    bitrep = bitdecomp(flat, n_bits)
    uint8rep = bitrecon(bitrep, n_bits, 8)

    return uint8rep.byte()

def unpack(p, n_bits, size=None):
    bitrep = bitdecomp(p, 8)
    recon = bitrecon(bitrep, 8, n_bits).short()

    if size is not None:
        nelements = np.prod(size)
        recon = recon[:nelements].view(size)

    return recon

if __name__ == '__main__':
    idx_high = 128
    a = torch.randint(low=0, high=idx_high, size=(4, 3)).long()
    p = numpack(a, int(math.log2(idx_high)))
    r = unpack(p, int(math.log2(idx_high)), a.size())
    diff = (a.short() - r).float().norm()

    print('Reconstruction error: {:.2f}'.format(diff))

