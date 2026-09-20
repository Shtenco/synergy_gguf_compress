#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
namespace gguf_compress_v3 {
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t s=(uint32_t(h&0x8000u))<<16; uint32_t e=(h>>10)&0x1fu; uint32_t f=h&0x03ffu; uint32_t out;
    if(e==0){ if(f==0){out=s;} else {e=1; while((f&0x0400u)==0){f<<=1;--e;} f&=0x03ffu; out=s|((e+(127-15))<<23)|(f<<13);}}
    else if(e==31){out=s|0x7f800000u|(f<<13);} else {out=s|((e+(127-15))<<23)|(f<<13);}
    float v; std::memcpy(&v,&out,sizeof(v)); return v;
}
inline int q4_at(const int8_t * packed,int col){const uint8_t byte=reinterpret_cast<const uint8_t*>(packed)[col>>1]; const uint8_t nibble=(col&1)?(byte>>4):(byte&0x0f); return int(nibble)-8;}
struct SparseResidualView {const int32_t* cols=nullptr; const int8_t* values=nullptr; const uint16_t* scales_f16=nullptr; int out_features=0; int k_per_row=0;};
inline void direct_group_q4(
        const float * x,
        int batch,
        int in_features,
        int out_features,
        int group_size,
        const int8_t * q4_data,
        const uint16_t * scales_f16,
        float * y) {
    const int packed_stride = (in_features + 1) / 2;
    const int groups = (in_features + group_size - 1) / group_size;
    for (int n = 0; n < batch; ++n) {
        const float * xn = x + size_t(n) * in_features;
        float * yn = y + size_t(n) * out_features;
        for (int o = 0; o < out_features; ++o) {
            const int8_t * row = q4_data + size_t(o) * packed_stride;
            const uint16_t * row_scales = scales_f16 + size_t(o) * groups;
            float sum = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                const float scale = fp16_to_fp32(row_scales[i / group_size]);
                sum += xn[i] * float(q4_at(row, i)) * scale;
            }
            yn[o] = sum;
        }
    }
}

inline void lowrank_q4_fused(const float* x,int batch,int in_features,int out_features,int rank,const int8_t* a_q,const uint16_t* a_scales_f16,const int8_t* b_q,const uint16_t* b_scales_f16,float* y,const SparseResidualView* residual=nullptr){
 const int a_stride=(rank+1)/2,b_stride=(in_features+1)/2; std::vector<float> hidden(rank);
 for(int n=0;n<batch;++n){const float* xn=x+size_t(n)*in_features;
  for(int r=0;r<rank;++r){const int8_t* brow=b_q+size_t(r)*b_stride; float sum=0; for(int i=0;i<in_features;++i) sum+=xn[i]*float(q4_at(brow,i)); hidden[r]=sum*fp16_to_fp32(b_scales_f16[r]);}
  float* yn=y+size_t(n)*out_features;
  for(int o=0;o<out_features;++o){const int8_t* arow=a_q+size_t(o)*a_stride; float sum=0; for(int r=0;r<rank;++r) sum+=hidden[r]*float(q4_at(arow,r)); yn[o]=sum*fp16_to_fp32(a_scales_f16[o]);}
  if(residual&&residual->cols&&residual->values&&residual->scales_f16){for(int o=0;o<out_features;++o){float scale=fp16_to_fp32(residual->scales_f16[o]); size_t base=size_t(o)*residual->k_per_row; float correction=0; for(int k=0;k<residual->k_per_row;++k) correction+=xn[residual->cols[base+k]]*float(residual->values[base+k]); yn[o]+=correction*scale;}}
 }}
}