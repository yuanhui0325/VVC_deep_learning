/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2020, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     DecCu.cpp
    \brief    CU decoder class
*/

#include "DecCu.h"

#include "CommonLib/CrossCompPrediction.h"
#include "CommonLib/InterPrediction.h"
#include "CommonLib/IntraPrediction.h"
#include "CommonLib/Picture.h"
#include "CommonLib/UnitTools.h"

#include "CommonLib/dtrace_buffer.h"

#if RExt__DECODER_DEBUG_TOOL_STATISTICS
#include "CommonLib/CodingStatistics.h"
#endif
#if K0149_BLOCK_STATISTICS
#include "CommonLib/ChromaFormat.h"
#include "CommonLib/dtrace_blockstatistics.h"
#endif
#if LIUYAO_TEST
#include <torch/script.h>
#include <vector>
#endif
#include<iostream>
#include <stdio.h>

//! \ingroup DecoderLib
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

DecCu::DecCu()
{
  m_tmpStorageLCU = NULL;
}

DecCu::~DecCu()
{
}

void DecCu::init( TrQuant* pcTrQuant, IntraPrediction* pcIntra, InterPrediction* pcInter)
{
  m_pcTrQuant       = pcTrQuant;
  m_pcIntraPred     = pcIntra;
  m_pcInterPred     = pcInter;
}
void DecCu::initDecCuReshaper  (Reshape* pcReshape, ChromaFormat chromaFormatIDC)
{
  m_pcReshape = pcReshape;
  if (m_tmpStorageLCU == NULL)
  {
    m_tmpStorageLCU = new PelStorage;
    m_tmpStorageLCU->create(UnitArea(chromaFormatIDC, Area(0, 0, MAX_CU_SIZE, MAX_CU_SIZE)));
  }

}
void DecCu::destoryDecCuReshaprBuf()
{
  if (m_tmpStorageLCU)
  {
    m_tmpStorageLCU->destroy();
    delete m_tmpStorageLCU;
    m_tmpStorageLCU = NULL;
  }
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================
int  sum1 = 0;
int sumANNM1 = 0;
void DecCu::decompressCtu( 
 #if LIUYAO_TEST
  int *reconParams,
  const unsigned ctuRsAddr,
#endif
  CodingStructure &cs, const UnitArea &ctuArea)
{

  const int maxNumChannelType = cs.pcv->chrFormat != CHROMA_400 && CS::isDualITree( cs ) ? 2 : 1;

  if (cs.resetIBCBuffer)
  {
    m_pcInterPred->resetIBCBuffer(cs.pcv->chrFormat, cs.slice->getSPS()->getMaxCUHeight());
    cs.resetIBCBuffer = false;
  }
  for( int ch = 0; ch < maxNumChannelType; ch++ )
  {
    const ChannelType chType = ChannelType( ch );
    Position prevTmpPos;
    prevTmpPos.x = -1; prevTmpPos.y = -1;

    for( auto &currCU : cs.traverseCUs( CS::getArea( cs, ctuArea, chType ), chType ) )
    {
      if(currCU.Y().valid())
      {
        const int vSize = cs.slice->getSPS()->getMaxCUHeight() > 64 ? 64 : cs.slice->getSPS()->getMaxCUHeight();
        if((currCU.Y().x % vSize) == 0 && (currCU.Y().y % vSize) == 0)
        {
          for(int x = currCU.Y().x; x < currCU.Y().x + currCU.Y().width; x += vSize)
          {
            for(int y = currCU.Y().y; y < currCU.Y().y + currCU.Y().height; y += vSize)
            {
              m_pcInterPred->resetVPDUforIBC(cs.pcv->chrFormat, cs.slice->getSPS()->getMaxCUHeight(), vSize, x + g_IBCBufferSize / cs.slice->getSPS()->getMaxCUHeight() / 2, y);
            }
          }
        }
      }
      if (currCU.predMode != MODE_INTRA && currCU.predMode != MODE_PLT && currCU.Y().valid())
      {
        xDeriveCUMV(currCU);
#if K0149_BLOCK_STATISTICS
        if(currCU.geoFlag)
        {
          storeGeoMergeCtx(m_geoMrgCtx);
        }
#endif
      }
      switch( currCU.predMode )
      {
      case MODE_INTER:
      case MODE_IBC:
        xReconInter( currCU );
        break;
      case MODE_PLT:
      case MODE_INTRA:
        xReconIntraQT( currCU );
        break;
      default:
        THROW( "Invalid prediction mode" );
        break;
      }

      m_pcInterPred->xFillIBCBuffer(currCU);

      DTRACE_BLOCK_REC( cs.picture->getRecoBuf( currCU ), currCU, currCU.predMode );
    }
  }
#if LIUYAO_TEST


  const int width  =cs.pcv->maxCUWidth;
  const int height = cs.pcv->maxCUHeight;

  bool flag      = false;
  bool flag_line = true;
  Pel *img_CTU_CNN[2];
  Pel *img_CTU_CNN_luma[1];  
  {
    img_CTU_CNN_luma[0] = new Pel[int(width) * int(height)];
  }
  for (int k = 0; k < 2; k++)
  {
    img_CTU_CNN[k] = new Pel[int(width / 2) * int(height / 2)];
  }
  Position          pos_L         = ctuArea.lumaPos();
  const CodingUnit *cuLeft_L      = cs.getCU(pos_L.offset(-width, 0), CH_L);
  const CodingUnit *cuAbove_L     =cs.getCU(pos_L.offset(0, -height), CH_L);
  const CodingUnit *cuAboveLeft_L =cs.getCU(pos_L.offset(-width, -height), CH_L);
  const CodingUnit *cuCurr_L      = cs.getCU(pos_L, CH_L);
  Position          pos_C    = ctuArea.chromaPos();
  const CodingUnit *cuCurr_C = cs.getCU(pos_C, CH_C);

  const CodingUnit *cuLeft_C      = cs.getCU(pos_C.offset(-int(width / 2), 0), CH_C);
  const CodingUnit *cuAbove_C     = cs.getCU(pos_C.offset(0, -int(height / 2)), CH_C);
  const CodingUnit *cuAboveLeft_C = cs.getCU(pos_C.offset(-int(width / 2), -int(height / 2)), CH_C);

  const int pic_width  =cs.pcv->lumaWidth;
  const int pic_height = cs.pcv->lumaHeight;

  // cout << pic_width << "," << pic_height << endl;

  const int uiLPelX = ctuArea.lumaPos().x;
  const int uiRPelX = uiLPelX + width - 1;
  const int uiTPelY = ctuArea.lumaPos().y;
  const int uiBPelY = uiTPelY + height - 1;
  if (uiRPelX > pic_width || uiBPelY > pic_height)
  {
    flag_line = false;
  }

  double maxvalue10 = 255.0;
  // double maxvalue8  = 255.0;

  //============================================================================================//
  // pytorch:准备神经网络的输入变量
  //============================================================================================//
  if (flag_line)
  {
    ////////////////////////////////////////////////////
    std::vector<float> refenceLuma_Above_vec;
    std::vector<float> refenceLuma_AboveLeft_vec;
    std::vector<float> refenceLuma_Left_vec;
    std::vector<float> refenceLuma_cur_vec;
    std::vector<float> refenceCb_Above_vec;
    std::vector<float> refenceCb_AboveLeft_vec;
    std::vector<float> refenceCb_Left_vec;
    std::vector<float> refenceCb_cur_vec;
    std::vector<float> refenceCr_Above_vec;
    std::vector<float> refenceCr_AboveLeft_vec;
    std::vector<float> refenceCr_Left_vec;
    std::vector<float> refenceCr_cur_vec;
    torch::Tensor      refenceCr_cur_tensor;
    torch::Tensor      refenceCb_cur_tensor;
    torch::Tensor      refenceLuma_cur_tensor;
    std::vector<float> QP_vec;
    torch::Tensor      luma;
    torch::Tensor      Cb;
    torch::Tensor      Cr;
    torch::Tensor      QP;
    /*
      {   // Y COMPONETNT
      PelBuf fig1 = cs.getRecoBuf(cuAbove_L->block(COMPONENT_Y));
      PelBuf fig0 = cs.getRecoBuf(cuAboveLeft_L->block(COMPONENT_Y));
      PelBuf fig2 = cs.getRecoBuf(cuLeft_L->block(COMPONENT_Y));
      //PelBuf fig3 = cs.getRecoBuf(cuCurr_L->block(COMPONENT_Y));

      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          refenceLuma_Above_vec.push_back(fig1.at(j, i) / maxvalue10);
          refenceLuma_AboveLeft_vec.push_back(fig0.at(j, i) / maxvalue10);
          refenceLuma_Left_vec.push_back(fig2.at(j, i) / maxvalue10);
          refenceLuma_cur_vec.push_back(fig3.at(j, i) / maxvalue10);
        }
      }
      torch::Tensor refenceLuma_Above_tensor     = torch::tensor(refenceLuma_Above_vec);
      refenceLuma_Above_tensor                   = refenceLuma_Above_tensor.view({ 1, 1, width, height });
      torch::Tensor refenceLuma_AboveLeft_tensor = torch::tensor(refenceLuma_AboveLeft_vec);
      refenceLuma_AboveLeft_tensor               = refenceLuma_AboveLeft_tensor.view({ 1, 1, width, height });
      torch::Tensor refenceLuma_Left_tensor      = torch::tensor(refenceLuma_Left_vec);
      refenceLuma_Left_tensor                    = refenceLuma_Left_tensor.view({ 1, 1, width, height });
      torch::Tensor refenceLuma_cur_tensor       = torch::tensor(refenceLuma_cur_vec);
      refenceLuma_cur_tensor                     = refenceLuma_cur_tensor.view({ 1, 1, width, height });
      torch::Tensor above = torch::cat({ refenceLuma_AboveLeft_tensor, refenceLuma_Above_tensor }, 2);
      torch::Tensor down  = torch::cat({ refenceLuma_Left_tensor, refenceLuma_cur_tensor }, 2);
      luma                = torch::cat({ above, down }, 3);
    }*/
    {   // Y COMPONETNT

      PelBuf fig3 = cs.getRecoBuf(cuCurr_L->block(COMPONENT_Y));

      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          refenceLuma_cur_vec.push_back(fig3.at(j, i) / maxvalue10);
          //img_CTU_CNN_luma[0][i * height + j] = fig3.at(j, i);
        }
      }

      refenceLuma_cur_tensor = torch::tensor(refenceLuma_cur_vec);
      refenceLuma_cur_tensor               = refenceLuma_cur_tensor.view({ 1, 1, width, height });
    }

    {   // CB COMPONETNT

      PelBuf fig3 = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cb));

      for (int i = 0; i < int(height / 2); i++)
      {
        for (int j = 0; j < int(width / 2); j++)
        {
          refenceCb_cur_vec.push_back(fig3.at(j, i) / maxvalue10);

        }
      }
      refenceCb_cur_tensor = torch::tensor(refenceCb_cur_vec);
      refenceCb_cur_tensor = refenceCb_cur_tensor.view({ 1, 1, width / 2, height / 2 });
    }

    {   // CR COMPONETNT

      PelBuf fig3 = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cr));

      for (int i = 0; i < int(height / 2); i++)
      {
        for (int j = 0; j < int(width / 2); j++)
        {
          refenceCr_cur_vec.push_back(fig3.at(j, i) / maxvalue10);
        }
      }
      refenceCr_cur_tensor = torch::tensor(refenceCr_cur_vec);
      refenceCr_cur_tensor = refenceCr_cur_tensor.view({ 1, 1, width / 2, height / 2 });
    }
    /*
        torch::Tensor cb_cr = torch::cat({ refenceCb_cur_tensor, refenceCr_cur_tensor }, 1);
    // refrence_yuv = torch::cat({ refrence_y, refrence_u, refrence_v }, 1);   //注意语法，要合并的tensor使用{}
    torch::jit::script::Module module;   //    torch::jit::script::Module module;
    module                 = torch::jit::load("G:/idea2_netparam/VDSR_76th.pt");
    at::Tensor output      = module.forward({ cb_cr }).toTensor();
    at::Tensor True_output = (output) *maxvalue10;

    for (int k = 0; k < 2; k++)
    {
      for (int uiY = 0; uiY < int(width / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(height / 2); uiX++)
        {
          img_CTU_CNN[k][uiY * int(height / 2) + uiX] = True_output[0][k][uiY][uiX].item().toShort();
        }
      }
    }*/
    {   // QP COMPONETNT
      //int qp0 = cs.baseQP;
      int qp0 =*cs.picture->m_prevQP;
      for (int i = 0; i < width; i++)
      {
        for (int j = 0; j < height; j++)
        {
          //int b = qp0 / 63.0;
          QP_vec.push_back(qp0 / 63.0);
        }
      }
      QP = torch::tensor(QP_vec);
      QP = QP.view({ 1, 1, width, height });
    }
    //torch::Tensor              luma_decoder  = torch::cat({ refenceCb_cur_tensor, refenceCr_cur_tensor }, 1);
    //torch::Tensor input = torch::cat({ refenceLuma_cur_tensor, QP }, 1);   //注意语法，要合并的tensor使用{}
    torch::jit::script::Module module;                                //    torch::jit::script::Module module;
    module                 = torch::jit::load("G:/idea2_netparam/MSR_NA_89th.pt");
    at::Tensor output      = module.forward({ refenceLuma_cur_tensor, QP }).toTensor();
    at::Tensor True_output = output *maxvalue10;

    for (int uiY = 0; uiY < width; uiY++)
    {
      for (int uiX = 0; uiX < height; uiX++)
      {
        //int a=True_output[0][0][uiY][uiX].item<int>();
       //td::cout<<True_output[0][0][uiY][uiX].item().toShort();
        //img_CTU_CNN_luma[0][uiY * height + uiX] = std::max(int(0), std::min(True_output[0][0][uiY][uiX].item<int>(), int(255)));
        //img_CTU_CNN_luma[0][uiY * height + uiX] =std::max(int16_t(0), std::min(True_output[0][0][uiY][uiX].item().toShort(), int16_t(255)));
       //img_CTU_CNN_luma[0][uiY * height + uiX] = True_output[0][0][uiY][uiX].item().toShort();
       img_CTU_CNN_luma[0][uiY * height + uiX] =std::max(int16_t(0), std::min(True_output[0][0][uiY][uiX].item().toShort(), int16_t(255)));
      }
    }

  }
  /*  if (flag_line)
  {
   
    sum1++;
    if (reconParams[ctuRsAddr] == 1)
    {
      sumANNM1++;
     
      const CodingUnit *cuCurr_C_CNN =cs.getCU(pos_C, CH_C);
      PelBuf            Cb_recCNN    = cs.getRecoBuf(cuCurr_C_CNN->block(COMPONENT_Cb));
      PelBuf            Cr_recCNN    = cs.getRecoBuf(cuCurr_C_CNN->block(COMPONENT_Cr));
      for (int k = 0; k < 2; k++)
      {
        for (int uiY = 0; uiY < int(height / 2); uiY++)
        {
          for (int uiX = 0; uiX < int(width / 2); uiX++)
          {
            if (k == 0)
            {
              Cb_recCNN.at(uiX, uiY) = img_CTU_CNN[0][uiY * int(height / 2) + uiX];
            }
            else
            {
              Cr_recCNN.at(uiX, uiY) = img_CTU_CNN[1][uiY * int(height / 2) + uiX];
            }
          }
        }
      }
    }
    printf("sum:%d,sum_ANNM:%d\n", sum1, sumANNM1);
#if TrueValue
    if (flag)
    {
      const CodingUnit *cuCurr_C_CNN = bestCS->getCU(pos_C, CH_C);
      PelBuf            fig_cur_Cb   = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cb));
      PelBuf            fig_cur_Cr   = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            Cb_recCNN    = cs.getRecoBuf(cuCurr_C_CNN->block(COMPONENT_Cb));
      PelBuf            Cr_recCNN    = cs.getRecoBuf(cuCurr_C_CNN->block(COMPONENT_Cr));
      for (int k = 0; k < 2; k++)
      {
        for (int uiY = 0; uiY < int(height / 2); uiY++)
        {
          for (int uiX = 0; uiX < int(width / 2); uiX++)
          {
            if (k == 0)
            {
              Cb_recCNN.at(uiX, uiY) = fig_cur_Cb.at(uiX, uiY);
            }
            else
            {
              Cr_recCNN.at(uiX, uiY) = fig_cur_Cr.at(uiX, uiY);
            }
          }
        }
      }
    }
#endif
#if 0
    {
      const CodingUnit *cuCurr_C       = bestCS->getCU(pos_C, CH_C);
      PelBuf            fig_cur_Cb     = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cb));
      PelBuf            fig_cur_Cr     = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            fig_cur_Cr_ped = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            fig_cur_Cb_ped = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cb));
      double            U_MSEAVE       = 0;
      double            U_PSNR_Rec     = 0;
      double            V_MSEAVE       = 0;
      double            V_PSNR_Rec     = 0;
      for (int uiY = 0; uiY < int(height / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(width / 2); uiX++)
        {
          double a = (fig_cur_Cb.at(uiX, uiY) - fig_cur_Cb_ped.at(uiX, uiY))
                     * (fig_cur_Cb.at(uiX, uiY) - fig_cur_Cb_ped.at(uiX, uiY));

          U_MSEAVE = U_MSEAVE + a / 4096.0;
        }
      }
      U_PSNR_Rec = 10 * log10((maxvalue10 * maxvalue10) / U_MSEAVE);
      for (int uiY = 0; uiY < int(height / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(width / 2); uiX++)
        {
          double a = (fig_cur_Cr.at(uiX, uiY) - fig_cur_Cr_ped.at(uiX, uiY))
                     * (fig_cur_Cr.at(uiX, uiY) - fig_cur_Cr_ped.at(uiX, uiY));

          V_MSEAVE = V_MSEAVE + a / 4096.0;
        }
      }
      V_PSNR_Rec = 10 * log10((maxvalue10 * maxvalue10) / V_MSEAVE);

      // printf("CNN增强重建_真实值：%f , %f\n", U_PSNR_Rec, V_PSNR_Rec);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////
      //使用网络增强CTU UV分量的预测准确度。
    }
#endif
  }
#endif*/
  if (flag_line)
  {
   
    sum1++;
    if (reconParams[ctuRsAddr] == 1)
    {
      sumANNM1++;
     
      const CodingUnit *cuCurr_Y_CNN =cs.getCU(pos_L, CH_L);
      PelBuf            Y_recCNN    = cs.getRecoBuf(cuCurr_Y_CNN->block(COMPONENT_Y));
      
      
        for (int uiY = 0; uiY < height ; uiY++)
        {
          for (int uiX = 0; uiX < width ; uiX++)
          {
           Y_recCNN.at(uiX, uiY) = img_CTU_CNN_luma[0][uiY * height + uiX];
            //printf("%d  ", img_CTU_CNN_luma[0][uiY * height + uiX]);
            //Y_recCNN.at(uiX, uiY) = 0;
           
          }
        }
      
    }
    printf("QP:%f\n", *cs.picture->m_prevQP / 63.0);
    printf("(x:%d,y:%d)\n", ctuArea.lumaPos().x, ctuArea.lumaPos().y);
    printf("sum:%d,sum_ANNM:%d\n", sum1, sumANNM1);
#if TrueValue
    if (flag)
    {
      const CodingUnit *cuCurr_C_CNN = bestCS->getCU(pos_C, CH_C);
      PelBuf            fig_cur_Cb   = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cb));
      PelBuf            fig_cur_Cr   = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            Cb_recCNN    = cs.getRecoBuf(cuCurr_C_CNN->block(COMPONENT_Cb));
      PelBuf            Cr_recCNN    = cs.getRecoBuf(cuCurr_C_CNN->block(COMPONENT_Cr));
      for (int k = 0; k < 2; k++)
      {
        for (int uiY = 0; uiY < int(height / 2); uiY++)
        {
          for (int uiX = 0; uiX < int(width / 2); uiX++)
          {
            if (k == 0)
            {
              Cb_recCNN.at(uiX, uiY) = fig_cur_Cb.at(uiX, uiY);
            }
            else
            {
              Cr_recCNN.at(uiX, uiY) = fig_cur_Cr.at(uiX, uiY);
            }
          }
        }
      }
    }
#endif
#if 0
    {
      const CodingUnit *cuCurr_C       = bestCS->getCU(pos_C, CH_C);
      PelBuf            fig_cur_Cb     = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cb));
      PelBuf            fig_cur_Cr     = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            fig_cur_Cr_ped = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            fig_cur_Cb_ped = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cb));
      double            U_MSEAVE       = 0;
      double            U_PSNR_Rec     = 0;
      double            V_MSEAVE       = 0;
      double            V_PSNR_Rec     = 0;
      for (int uiY = 0; uiY < int(height / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(width / 2); uiX++)
        {
          double a = (fig_cur_Cb.at(uiX, uiY) - fig_cur_Cb_ped.at(uiX, uiY))
                     * (fig_cur_Cb.at(uiX, uiY) - fig_cur_Cb_ped.at(uiX, uiY));

          U_MSEAVE = U_MSEAVE + a / 4096.0;
        }
      }
      U_PSNR_Rec = 10 * log10((maxvalue10 * maxvalue10) / U_MSEAVE);
      for (int uiY = 0; uiY < int(height / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(width / 2); uiX++)
        {
          double a = (fig_cur_Cr.at(uiX, uiY) - fig_cur_Cr_ped.at(uiX, uiY))
                     * (fig_cur_Cr.at(uiX, uiY) - fig_cur_Cr_ped.at(uiX, uiY));

          V_MSEAVE = V_MSEAVE + a / 4096.0;
        }
      }
      V_PSNR_Rec = 10 * log10((maxvalue10 * maxvalue10) / V_MSEAVE);

      // printf("CNN增强重建_真实值：%f , %f\n", U_PSNR_Rec, V_PSNR_Rec);
      ////////////////////////////////////////////////////////////////////////////////////////////////////////
      //使用网络增强CTU UV分量的预测准确度。
    }
#endif
  }
#endif

#if K0149_BLOCK_STATISTICS
  getAndStoreBlockStatistics(cs, ctuArea);
#endif
}

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

void DecCu::xIntraRecBlk( TransformUnit& tu, const ComponentID compID )
{
  if( !tu.blocks[ compID ].valid() )
  {
    return;
  }

        CodingStructure &cs = *tu.cs;
  const CompArea &area      = tu.blocks[compID];

  const ChannelType chType  = toChannelType( compID );

        PelBuf piPred       = cs.getPredBuf( area );

  const PredictionUnit &pu  = *tu.cs->getPU( area.pos(), chType );
  const uint32_t uiChFinalMode  = PU::getFinalIntraMode( pu, chType );
  PelBuf pReco              = cs.getRecoBuf(area);

  //===== init availability pattern =====
  bool predRegDiffFromTB = CU::isPredRegDiffFromTB(*tu.cu, compID);
  bool firstTBInPredReg = CU::isFirstTBInPredReg(*tu.cu, compID, area);
  CompArea areaPredReg(COMPONENT_Y, tu.chromaFormat, area);
  if (tu.cu->ispMode && isLuma(compID))
  {
    if (predRegDiffFromTB)
    {
      if (firstTBInPredReg)
      {
        CU::adjustPredArea(areaPredReg);
        m_pcIntraPred->initIntraPatternChTypeISP(*tu.cu, areaPredReg, pReco);
      }
    }
    else
    {
      m_pcIntraPred->initIntraPatternChTypeISP(*tu.cu, area, pReco);
    }
  }
  else
  {
    m_pcIntraPred->initIntraPatternChType(*tu.cu, area);
  }

  //===== get prediction signal =====
  if( compID != COMPONENT_Y && PU::isLMCMode( uiChFinalMode ) )
  {
    const PredictionUnit& pu = *tu.cu->firstPU;
    m_pcIntraPred->xGetLumaRecPixels( pu, area );
    m_pcIntraPred->predIntraChromaLM( compID, piPred, pu, area, uiChFinalMode );
  }
  else
  {
    if( PU::isMIP( pu, chType ) )
    {
      m_pcIntraPred->initIntraMip( pu, area );
      m_pcIntraPred->predIntraMip( compID, piPred, pu );
    }
    else
    {
      if (predRegDiffFromTB)
      {
        if (firstTBInPredReg)
        {
          PelBuf piPredReg = cs.getPredBuf(areaPredReg);
          m_pcIntraPred->predIntraAng(compID, piPredReg, pu);
        }
      }
      else
        m_pcIntraPred->predIntraAng(compID, piPred, pu);
    }
  }
  const Slice           &slice = *cs.slice;
  bool flag = slice.getLmcsEnabledFlag() && (slice.isIntra() || (!slice.isIntra() && m_pcReshape->getCTUFlag()));
  if (flag && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && (compID != COMPONENT_Y) && (tu.cbf[COMPONENT_Cb] || tu.cbf[COMPONENT_Cr]))
  {
    const Area area = tu.Y().valid() ? tu.Y() : Area(recalcPosition(tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, tu.blocks[tu.chType].pos()), recalcSize(tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, tu.blocks[tu.chType].size()));
    const CompArea &areaY = CompArea(COMPONENT_Y, tu.chromaFormat, area);
    int adj = m_pcReshape->calculateChromaAdjVpduNei(tu, areaY);
    tu.setChromaAdj(adj);
  }
  //===== inverse transform =====
  PelBuf piResi = cs.getResiBuf( area );

  const QpParam cQP( tu, compID );

  if( tu.jointCbCr && isChroma(compID) )
  {
    if( compID == COMPONENT_Cb )
    {
      PelBuf resiCr = cs.getResiBuf( tu.blocks[ COMPONENT_Cr ] );
      if( tu.jointCbCr >> 1 )
      {
        m_pcTrQuant->invTransformNxN( tu, COMPONENT_Cb, piResi, cQP );
      }
      else
      {
        const QpParam qpCr( tu, COMPONENT_Cr );
        m_pcTrQuant->invTransformNxN( tu, COMPONENT_Cr, resiCr, qpCr );
      }
      m_pcTrQuant->invTransformICT( tu, piResi, resiCr );
    }
  }
  else
  if( TU::getCbf( tu, compID ) )
  {
    m_pcTrQuant->invTransformNxN( tu, compID, piResi, cQP );
  }
  else
  {
    piResi.fill( 0 );
  }

  //===== reconstruction =====
  flag = flag && (tu.blocks[compID].width*tu.blocks[compID].height > 4);
  if (flag && (TU::getCbf(tu, compID) || tu.jointCbCr) && isChroma(compID) && slice.getPicHeader()->getLmcsChromaResidualScaleFlag())
  {
    piResi.scaleSignal(tu.getChromaAdj(), 0, tu.cu->cs->slice->clpRng(compID));
  }

  if( !tu.cu->ispMode || !isLuma( compID ) )
  {
    cs.setDecomp( area );
  }
  else if( tu.cu->ispMode && isLuma( compID ) && CU::isISPFirst( *tu.cu, tu.blocks[compID], compID ) )
  {
    cs.setDecomp( tu.cu->blocks[compID] );
  }

#if REUSE_CU_RESULTS
  CompArea    tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
  PelBuf tmpPred;
#endif
  if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
  {
#if REUSE_CU_RESULTS
    {
      tmpPred = m_tmpStorageLCU->getBuf(tmpArea);
      tmpPred.copyFrom(piPred);
    }
#endif
  }
#if KEEP_PRED_AND_RESI_SIGNALS
  pReco.reconstruct( piPred, piResi, tu.cu->cs->slice->clpRng( compID ) );
#else
  piPred.reconstruct( piPred, piResi, tu.cu->cs->slice->clpRng( compID ) );
#endif
#if !KEEP_PRED_AND_RESI_SIGNALS
  pReco.copyFrom( piPred );
#endif
  if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
  {
#if REUSE_CU_RESULTS
    {
      piPred.copyFrom(tmpPred);
    }
#endif
  }
#if REUSE_CU_RESULTS
  if( cs.pcv->isEncoder )
  {
    cs.picture->getRecoBuf( area ).copyFrom( pReco );
    cs.picture->getPredBuf(area).copyFrom(piPred);
  }
#endif
}

void DecCu::xIntraRecACTBlk(TransformUnit& tu)
{
  CodingStructure      &cs = *tu.cs;
  const PredictionUnit &pu = *tu.cs->getPU(tu.blocks[COMPONENT_Y], CHANNEL_TYPE_LUMA);
  const Slice          &slice = *cs.slice;

  CHECK_VTM(!tu.Y().valid() || !tu.Cb().valid() || !tu.Cr().valid(), "Invalid TU");
  CHECK_VTM(&pu != tu.cu->firstPU, "wrong PU fetch");
  CHECK_VTM(tu.cu->ispMode, "adaptive color transform cannot be applied to ISP");
  CHECK_VTM(pu.intraDir[CHANNEL_TYPE_CHROMA] != DM_CHROMA_IDX, "chroma should use DM mode for adaptive color transform");

  bool flag = slice.getLmcsEnabledFlag() && (slice.isIntra() || (!slice.isIntra() && m_pcReshape->getCTUFlag()));
  if (flag && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && (tu.cbf[COMPONENT_Cb] || tu.cbf[COMPONENT_Cr]))
  {
    const Area      area = tu.Y().valid() ? tu.Y() : Area(recalcPosition(tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, tu.blocks[tu.chType].pos()), recalcSize(tu.chromaFormat, tu.chType, CHANNEL_TYPE_LUMA, tu.blocks[tu.chType].size()));
    const CompArea &areaY = CompArea(COMPONENT_Y, tu.chromaFormat, area);
    int            adj = m_pcReshape->calculateChromaAdjVpduNei(tu, areaY);
    tu.setChromaAdj(adj);
  }

  for (int i = 0; i < getNumberValidComponents(tu.chromaFormat); i++)
  {
    ComponentID          compID = (ComponentID)i;
    const CompArea       &area = tu.blocks[compID];
    const ChannelType    chType = toChannelType(compID);

    PelBuf piPred = cs.getPredBuf(area);
    m_pcIntraPred->initIntraPatternChType(*tu.cu, area);
    if (PU::isMIP(pu, chType))
    {
      m_pcIntraPred->initIntraMip(pu, area);
      m_pcIntraPred->predIntraMip(compID, piPred, pu);
    }
    else
    {
      m_pcIntraPred->predIntraAng(compID, piPred, pu);
    }

    PelBuf piResi = cs.getResiBuf(area);

    QpParam cQP(tu, compID);

    if (tu.jointCbCr && isChroma(compID))
    {
      if (compID == COMPONENT_Cb)
      {
        PelBuf resiCr = cs.getResiBuf(tu.blocks[COMPONENT_Cr]);
        if (tu.jointCbCr >> 1)
        {
          m_pcTrQuant->invTransformNxN(tu, COMPONENT_Cb, piResi, cQP);
        }
        else
        {
          QpParam qpCr(tu, COMPONENT_Cr);

          m_pcTrQuant->invTransformNxN(tu, COMPONENT_Cr, resiCr, qpCr);
        }
        m_pcTrQuant->invTransformICT(tu, piResi, resiCr);
      }
    }
    else
    {
      if (TU::getCbf(tu, compID))
      {
        m_pcTrQuant->invTransformNxN(tu, compID, piResi, cQP);
      }
      else
      {
        piResi.fill(0);
      }
    }

    flag = flag && (tu.blocks[compID].width*tu.blocks[compID].height > 4);
    if (flag && (TU::getCbf(tu, compID) || tu.jointCbCr) && isChroma(compID) && slice.getPicHeader()->getLmcsChromaResidualScaleFlag())
    {
      piResi.scaleSignal(tu.getChromaAdj(), 0, tu.cu->cs->slice->clpRng(compID));
    }

    cs.setDecomp(area);
  }

  cs.getResiBuf(tu).colorSpaceConvert(cs.getResiBuf(tu), false, tu.cu->cs->slice->clpRng(COMPONENT_Y));

  for (int i = 0; i < getNumberValidComponents(tu.chromaFormat); i++)
  {
    ComponentID          compID = (ComponentID)i;
    const CompArea       &area = tu.blocks[compID];

    PelBuf piPred = cs.getPredBuf(area);
    PelBuf piResi = cs.getResiBuf(area);
    PelBuf piReco = cs.getRecoBuf(area);

    PelBuf tmpPred;
    if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
    {
      CompArea tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
      tmpPred = m_tmpStorageLCU->getBuf(tmpArea);
      tmpPred.copyFrom(piPred);
    }

    piPred.reconstruct(piPred, piResi, tu.cu->cs->slice->clpRng(compID));
    piReco.copyFrom(piPred);

    if (slice.getLmcsEnabledFlag() && (m_pcReshape->getCTUFlag() || slice.isIntra()) && compID == COMPONENT_Y)
    {
      piPred.copyFrom(tmpPred);
    }

    if (cs.pcv->isEncoder)
    {
      cs.picture->getRecoBuf(area).copyFrom(piReco);
      cs.picture->getPredBuf(area).copyFrom(piPred);
    }
  }
}

void DecCu::xReconIntraQT( CodingUnit &cu )
{

  if (CU::isPLT(cu))
  {
    if (cu.isSepTree())
    {
      if (cu.chType == CHANNEL_TYPE_LUMA)
      {
        xReconPLT(cu, COMPONENT_Y, 1);
      }
      if (cu.chromaFormat != CHROMA_400 && (cu.chType == CHANNEL_TYPE_CHROMA))
      {
        xReconPLT(cu, COMPONENT_Cb, 2);
      }
    }
    else
    {
      if( cu.chromaFormat != CHROMA_400 )
      {
        xReconPLT(cu, COMPONENT_Y, 3);
      }
      else
      {
        xReconPLT(cu, COMPONENT_Y, 1);
      }
    }
    return;
  }

  if (cu.colorTransform)
  {
    xIntraRecACTQT(cu);
  }
  else
  {
  const uint32_t numChType = ::getNumberValidChannels( cu.chromaFormat );

  for( uint32_t chType = CHANNEL_TYPE_LUMA; chType < numChType; chType++ )
  {
    if( cu.blocks[chType].valid() )
    {
      xIntraRecQT( cu, ChannelType( chType ) );
    }
  }
  }
}

void DecCu::xReconPLT(CodingUnit &cu, ComponentID compBegin, uint32_t numComp)
{
  const SPS&       sps = *(cu.cs->sps);
  TransformUnit&   tu = *cu.firstTU;
  PelBuf    curPLTIdx = tu.getcurPLTIdx(compBegin);

  uint32_t height = cu.block(compBegin).height;
  uint32_t width = cu.block(compBegin).width;

  //recon. pixels
  uint32_t scaleX = getComponentScaleX(COMPONENT_Cb, sps.getChromaFormatIdc());
  uint32_t scaleY = getComponentScaleY(COMPONENT_Cb, sps.getChromaFormatIdc());
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      for (uint32_t compID = compBegin; compID < (compBegin + numComp); compID++)
      {
        const int  channelBitDepth = cu.cs->sps->getBitDepth(toChannelType((ComponentID)compID));
        const CompArea &area = cu.blocks[compID];

        PelBuf       picReco   = cu.cs->getRecoBuf(area);
        PLTescapeBuf escapeValue = tu.getescapeValue((ComponentID)compID);
        if (curPLTIdx.at(x, y) == cu.curPLTSize[compBegin])
        {
          Pel value;
          QpParam cQP(tu, (ComponentID)compID);
          int qp = cQP.Qp(true);
          int qpRem = qp % 6;
          int qpPer = qp / 6;
          if (compBegin != COMPONENT_Y || compID == COMPONENT_Y)
          {
            int invquantiserRightShift = IQUANT_SHIFT;
            int add = 1 << (invquantiserRightShift - 1);
            value = ((((escapeValue.at(x, y)*g_invQuantScales[0][qpRem]) << qpPer) + add) >> invquantiserRightShift);
            value = Pel(ClipBD<int>(value, channelBitDepth));
            picReco.at(x, y) = value;
          }
          else if (compBegin == COMPONENT_Y && compID != COMPONENT_Y && y % (1 << scaleY) == 0 && x % (1 << scaleX) == 0)
          {
            uint32_t posYC = y >> scaleY;
            uint32_t posXC = x >> scaleX;
            int invquantiserRightShift = IQUANT_SHIFT;
            int add = 1 << (invquantiserRightShift - 1);
            value = ((((escapeValue.at(posXC, posYC)*g_invQuantScales[0][qpRem]) << qpPer) + add) >> invquantiserRightShift);
            value = Pel(ClipBD<int>(value, channelBitDepth));
            picReco.at(posXC, posYC) = value;

          }
        }
        else
        {
          uint32_t curIdx = curPLTIdx.at(x, y);
          if (compBegin != COMPONENT_Y || compID == COMPONENT_Y)
          {
            picReco.at(x, y) = cu.curPLT[compID][curIdx];
          }
          else if (compBegin == COMPONENT_Y && compID != COMPONENT_Y && y % (1 << scaleY) == 0 && x % (1 << scaleX) == 0)
          {
            uint32_t posYC = y >> scaleY;
            uint32_t posXC = x >> scaleX;
            picReco.at(posXC, posYC) = cu.curPLT[compID][curIdx];
          }
        }
      }
    }
  }
  for (uint32_t compID = compBegin; compID < (compBegin + numComp); compID++)
  {
    const CompArea &area = cu.blocks[compID];
    PelBuf picReco = cu.cs->getRecoBuf(area);
    cu.cs->picture->getRecoBuf(area).copyFrom(picReco);
    cu.cs->setDecomp(area);
  }
}

/** Function for deriving reconstructed PU/CU chroma samples with QTree structure
* \param pcRecoYuv pointer to reconstructed sample arrays
* \param pcPredYuv pointer to prediction sample arrays
* \param pcResiYuv pointer to residue sample arrays
* \param chType    texture channel type (luma/chroma)
* \param rTu       reference to transform data
*
\ This function derives reconstructed PU/CU chroma samples with QTree recursive structure
*/

void
DecCu::xIntraRecQT(CodingUnit &cu, const ChannelType chType)
{
  for( auto &currTU : CU::traverseTUs( cu ) )
  {
    if( isLuma( chType ) )
    {
      xIntraRecBlk( currTU, COMPONENT_Y );
    }
    else
    {
      const uint32_t numValidComp = getNumberValidComponents( cu.chromaFormat );

      for( uint32_t compID = COMPONENT_Cb; compID < numValidComp; compID++ )
      {
        xIntraRecBlk( currTU, ComponentID( compID ) );
      }
    }
  }
}

void DecCu::xIntraRecACTQT(CodingUnit &cu)
{
  for (auto &currTU : CU::traverseTUs(cu))
  {
    xIntraRecACTBlk(currTU);
  }
}

/** Function for filling the PCM buffer of a CU using its reconstructed sample array
* \param pCU   pointer to current CU
* \param depth CU Depth
*/
void DecCu::xFillPCMBuffer(CodingUnit &cu)
{
  for( auto &currTU : CU::traverseTUs( cu ) )
  {
    for (const CompArea &area : currTU.blocks)
    {
      if( !area.valid() ) continue;;

      CPelBuf source      = cu.cs->getRecoBuf(area);
       PelBuf destination = currTU.getPcmbuf(area.compID);

      destination.copyFrom(source);
    }
  }
}

#include "CommonLib/dtrace_buffer.h"

void DecCu::xReconInter(CodingUnit &cu)
{
  if( cu.geoFlag )
  {
    m_pcInterPred->motionCompensationGeo( cu, m_geoMrgCtx );
    PU::spanGeoMotionInfo( *cu.firstPU, m_geoMrgCtx, cu.firstPU->geoSplitDir, cu.firstPU->geoMergeIdx0, cu.firstPU->geoMergeIdx1 );
  }
  else
  {
  m_pcIntraPred->geneIntrainterPred(cu);

  // inter prediction
  CHECK_VTM(CU::isIBC(cu) && cu.firstPU->ciipFlag, "IBC and Ciip cannot be used together");
  CHECK_VTM(CU::isIBC(cu) && cu.affine, "IBC and Affine cannot be used together");
  CHECK_VTM(CU::isIBC(cu) && cu.geoFlag, "IBC and geo cannot be used together");
  CHECK_VTM(CU::isIBC(cu) && cu.firstPU->mmvdMergeFlag, "IBC and MMVD cannot be used together");
  const bool luma = cu.Y().valid();
  const bool chroma = isChromaEnabled(cu.chromaFormat) && cu.Cb().valid();
  if (luma && (chroma || !isChromaEnabled(cu.chromaFormat)))
  {
    m_pcInterPred->motionCompensation(cu);
  }
  else
  {
    m_pcInterPred->motionCompensation(cu, REF_PIC_LIST_0, luma, chroma);
  }
  }
  if (cu.Y().valid())
  {
    bool isIbcSmallBlk = CU::isIBC(cu) && (cu.lwidth() * cu.lheight() <= 16);
    CU::saveMotionInHMVP( cu, isIbcSmallBlk );
  }

  if (cu.firstPU->ciipFlag)
  {
    if (cu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
      cu.cs->getPredBuf(*cu.firstPU).Y().rspSignal(m_pcReshape->getFwdLUT());
    }
    m_pcIntraPred->geneWeightedPred(COMPONENT_Y, cu.cs->getPredBuf(*cu.firstPU).Y(), *cu.firstPU, m_pcIntraPred->getPredictorPtr2(COMPONENT_Y, 0));
    if (isChromaEnabled(cu.chromaFormat) && cu.chromaSize().width > 2)
    {
      m_pcIntraPred->geneWeightedPred(COMPONENT_Cb, cu.cs->getPredBuf(*cu.firstPU).Cb(), *cu.firstPU, m_pcIntraPred->getPredictorPtr2(COMPONENT_Cb, 0));
      m_pcIntraPred->geneWeightedPred(COMPONENT_Cr, cu.cs->getPredBuf(*cu.firstPU).Cr(), *cu.firstPU, m_pcIntraPred->getPredictorPtr2(COMPONENT_Cr, 0));
    }
  }

  DTRACE    ( g_trace_ctx, D_TMP, "pred " );
  DTRACE_CRC( g_trace_ctx, D_TMP, *cu.cs, cu.cs->getPredBuf( cu ), &cu.Y() );

  // inter recon
  xDecodeInterTexture(cu);

  // clip for only non-zero cbf case
  CodingStructure &cs = *cu.cs;

  if (cu.rootCbf)
  {
    if (cu.colorTransform)
    {
      cs.getResiBuf(cu).colorSpaceConvert(cs.getResiBuf(cu), false, cu.cs->slice->clpRng(COMPONENT_Y));
    }
#if REUSE_CU_RESULTS
    const CompArea &area = cu.blocks[COMPONENT_Y];
    CompArea    tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
    PelBuf tmpPred;
#endif
    if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
#if REUSE_CU_RESULTS
      if (cs.pcv->isEncoder)
      {
        tmpPred = m_tmpStorageLCU->getBuf(tmpArea);
        tmpPred.copyFrom(cs.getPredBuf(cu).get(COMPONENT_Y));
      }
#endif
      if (!cu.firstPU->ciipFlag && !CU::isIBC(cu))
        cs.getPredBuf(cu).get(COMPONENT_Y).rspSignal(m_pcReshape->getFwdLUT());
    }
#if KEEP_PRED_AND_RESI_SIGNALS
    cs.getRecoBuf( cu ).reconstruct( cs.getPredBuf( cu ), cs.getResiBuf( cu ), cs.slice->clpRngs() );
#else
    cs.getResiBuf( cu ).reconstruct( cs.getPredBuf( cu ), cs.getResiBuf( cu ), cs.slice->clpRngs() );
    cs.getRecoBuf( cu ).copyFrom   (                      cs.getResiBuf( cu ) );
#endif
    if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
    {
#if REUSE_CU_RESULTS
      if (cs.pcv->isEncoder)
      {
        cs.getPredBuf(cu).get(COMPONENT_Y).copyFrom(tmpPred);
      }
#endif
    }
  }
  else
  {
    cs.getRecoBuf(cu).copyClip(cs.getPredBuf(cu), cs.slice->clpRngs());
    if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag() && !cu.firstPU->ciipFlag && !CU::isIBC(cu))
    {
      cs.getRecoBuf(cu).get(COMPONENT_Y).rspSignal(m_pcReshape->getFwdLUT());
    }
  }

  DTRACE    ( g_trace_ctx, D_TMP, "reco " );
  DTRACE_CRC( g_trace_ctx, D_TMP, *cu.cs, cu.cs->getRecoBuf( cu ), &cu.Y() );

  cs.setDecomp(cu);
}

void DecCu::xDecodeInterTU( TransformUnit & currTU, const ComponentID compID )
{
  if( !currTU.blocks[compID].valid() ) return;

  const CompArea &area = currTU.blocks[compID];

  CodingStructure& cs = *currTU.cs;

  //===== inverse transform =====
  PelBuf resiBuf  = cs.getResiBuf(area);

  QpParam cQP(currTU, compID);

  if( currTU.jointCbCr && isChroma(compID) )
  {
    if( compID == COMPONENT_Cb )
    {
      PelBuf resiCr = cs.getResiBuf( currTU.blocks[ COMPONENT_Cr ] );
      if( currTU.jointCbCr >> 1 )
      {
        m_pcTrQuant->invTransformNxN( currTU, COMPONENT_Cb, resiBuf, cQP );
      }
      else
      {
        QpParam qpCr(currTU, COMPONENT_Cr);
        m_pcTrQuant->invTransformNxN( currTU, COMPONENT_Cr, resiCr, qpCr );
      }
      m_pcTrQuant->invTransformICT( currTU, resiBuf, resiCr );
    }
  }
  else
  if( TU::getCbf( currTU, compID ) )
  {
    m_pcTrQuant->invTransformNxN( currTU, compID, resiBuf, cQP );
  }
  else
  {
    resiBuf.fill( 0 );
  }

  //===== reconstruction =====
  const Slice           &slice = *cs.slice;
  if (slice.getLmcsEnabledFlag() && isChroma(compID) && (TU::getCbf(currTU, compID) || currTU.jointCbCr)
   && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && currTU.blocks[compID].width * currTU.blocks[compID].height > 4)
  {
    resiBuf.scaleSignal(currTU.getChromaAdj(), 0, currTU.cu->cs->slice->clpRng(compID));
  }
}

void DecCu::xDecodeInterTexture(CodingUnit &cu)
{
  if( !cu.rootCbf )
  {
    return;
  }

  const uint32_t uiNumVaildComp = getNumberValidComponents(cu.chromaFormat);

  for (uint32_t ch = 0; ch < uiNumVaildComp; ch++)
  {
    const ComponentID compID = ComponentID(ch);

    for( auto& currTU : CU::traverseTUs( cu ) )
    {
      CodingStructure  &cs = *cu.cs;
      const Slice &slice = *cs.slice;
      if (slice.getLmcsEnabledFlag() && slice.getPicHeader()->getLmcsChromaResidualScaleFlag() && (compID == COMPONENT_Y) && (currTU.cbf[COMPONENT_Cb] || currTU.cbf[COMPONENT_Cr]))
      {
        const CompArea &areaY = currTU.blocks[COMPONENT_Y];
        int adj = m_pcReshape->calculateChromaAdjVpduNei(currTU, areaY);
        currTU.setChromaAdj(adj);
    }
      xDecodeInterTU( currTU, compID );
    }
  }
}

void DecCu::xDeriveCUMV( CodingUnit &cu )
{
  for( auto &pu : CU::traversePUs( cu ) )
  {
    MergeCtx mrgCtx;

#if RExt__DECODER_DEBUG_TOOL_STATISTICS
    if( pu.cu->affine )
    {
      CodingStatistics::IncrementStatisticTool( CodingStatisticsClassType{ STATS__TOOL_AFF, pu.Y().width, pu.Y().height } );
    }
#endif


    if( pu.mergeFlag )
    {
      if (pu.mmvdMergeFlag || pu.cu->mmvdSkip)
      {
        CHECK_VTM(pu.ciipFlag == true, "invalid Ciip");
        if (pu.cs->sps->getSBTMVPEnabledFlag())
        {
          Size bufSize = g_miScaling.scale(pu.lumaSize());
          mrgCtx.subPuMvpMiBuf = MotionBuf(m_SubPuMiBuf, bufSize);
        }

        int   fPosBaseIdx = pu.mmvdMergeIdx / MMVD_MAX_REFINE_NUM;
        PU::getInterMergeCandidates(pu, mrgCtx, 1, fPosBaseIdx + 1);
        PU::getInterMMVDMergeCandidates(pu, mrgCtx,
          pu.mmvdMergeIdx
        );
        mrgCtx.setMmvdMergeCandiInfo(pu, pu.mmvdMergeIdx);

        PU::spanMotionInfo(pu, mrgCtx);
      }
      else
      {
      {
        if( pu.cu->geoFlag )
        {
          PU::getGeoMergeCandidates( pu, m_geoMrgCtx );
        }
        else
        {
        if( pu.cu->affine )
        {
          AffineMergeCtx affineMergeCtx;
          if ( pu.cs->sps->getSBTMVPEnabledFlag() )
          {
            Size bufSize = g_miScaling.scale( pu.lumaSize() );
            mrgCtx.subPuMvpMiBuf = MotionBuf( m_SubPuMiBuf, bufSize );
            affineMergeCtx.mrgCtx = &mrgCtx;
          }
          PU::getAffineMergeCand( pu, affineMergeCtx, pu.mergeIdx );
          pu.interDir = affineMergeCtx.interDirNeighbours[pu.mergeIdx];
          pu.cu->affineType = affineMergeCtx.affineType[pu.mergeIdx];
          pu.cu->BcwIdx = affineMergeCtx.BcwIdx[pu.mergeIdx];
          pu.mergeType = affineMergeCtx.mergeType[pu.mergeIdx];
          if ( pu.mergeType == MRG_TYPE_SUBPU_ATMVP )
          {
            pu.refIdx[0] = affineMergeCtx.mvFieldNeighbours[(pu.mergeIdx << 1) + 0][0].refIdx;
            pu.refIdx[1] = affineMergeCtx.mvFieldNeighbours[(pu.mergeIdx << 1) + 1][0].refIdx;
          }
          else
          {
          for( int i = 0; i < 2; ++i )
          {
            if( pu.cs->slice->getNumRefIdx( RefPicList( i ) ) > 0 )
            {
              MvField* mvField = affineMergeCtx.mvFieldNeighbours[(pu.mergeIdx << 1) + i];
              pu.mvpIdx[i] = 0;
              pu.mvpNum[i] = 0;
              pu.mvd[i]    = Mv();
              PU::setAllAffineMvField( pu, mvField, RefPicList( i ) );
            }
          }
        }
          PU::spanMotionInfo( pu, mrgCtx );
        }
        else
        {
          if (CU::isIBC(*pu.cu))
            PU::getIBCMergeCandidates(pu, mrgCtx, pu.mergeIdx);
          else
            PU::getInterMergeCandidates(pu, mrgCtx, 0, pu.mergeIdx);
          mrgCtx.setMergeInfo( pu, pu.mergeIdx );

          PU::spanMotionInfo( pu, mrgCtx );
        }
        }
      }
      }
    }
    else
    {
#if REUSE_CU_RESULTS
      if ( cu.imv && !pu.cu->affine && !cu.cs->pcv->isEncoder )
#else
        if (cu.imv && !pu.cu->affine)
#endif
        {
          PU::applyImv(pu, mrgCtx, m_pcInterPred);
        }
        else
      {
        if( pu.cu->affine )
        {
          for ( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            RefPicList eRefList = RefPicList( uiRefListIdx );
            if ( pu.cs->slice->getNumRefIdx( eRefList ) > 0 && ( pu.interDir & ( 1 << uiRefListIdx ) ) )
            {
              AffineAMVPInfo affineAMVPInfo;
              PU::fillAffineMvpCand( pu, eRefList, pu.refIdx[eRefList], affineAMVPInfo );

              const unsigned mvp_idx = pu.mvpIdx[eRefList];

              pu.mvpNum[eRefList] = affineAMVPInfo.numCand;

              //    Mv mv[3];
              CHECK_VTM( pu.refIdx[eRefList] < 0, "Unexpected negative refIdx." );
              if (!cu.cs->pcv->isEncoder)
              {
                pu.mvdAffi[eRefList][0].changeAffinePrecAmvr2Internal(pu.cu->imv);
                pu.mvdAffi[eRefList][1].changeAffinePrecAmvr2Internal(pu.cu->imv);
                if (cu.affineType == AFFINEMODEL_6PARAM)
                {
                  pu.mvdAffi[eRefList][2].changeAffinePrecAmvr2Internal(pu.cu->imv);
                }
              }

              Mv mvLT = affineAMVPInfo.mvCandLT[mvp_idx] + pu.mvdAffi[eRefList][0];
              Mv mvRT = affineAMVPInfo.mvCandRT[mvp_idx] + pu.mvdAffi[eRefList][1];
              mvRT += pu.mvdAffi[eRefList][0];

              Mv mvLB;
              if ( cu.affineType == AFFINEMODEL_6PARAM )
              {
                mvLB = affineAMVPInfo.mvCandLB[mvp_idx] + pu.mvdAffi[eRefList][2];
                mvLB += pu.mvdAffi[eRefList][0];
              }
              PU::setAllAffineMv(pu, mvLT, mvRT, mvLB, eRefList, true);
            }
          }
        }
        else if (CU::isIBC(*pu.cu) && pu.interDir == 1)
        {
          AMVPInfo amvpInfo;
          PU::fillIBCMvpCand(pu, amvpInfo);
          pu.mvpNum[REF_PIC_LIST_0] = amvpInfo.numCand;
          Mv mvd = pu.mvd[REF_PIC_LIST_0];
#if REUSE_CU_RESULTS
          if (!cu.cs->pcv->isEncoder)
#endif
          {
            mvd.changeIbcPrecAmvr2Internal(pu.cu->imv);
          }
          if (pu.cs->sps->getMaxNumIBCMergeCand() == 1)
          {
            CHECK_VTM( pu.mvpIdx[REF_PIC_LIST_0], "mvpIdx for IBC mode should be 0" );
          }
          pu.mv[REF_PIC_LIST_0] = amvpInfo.mvCand[pu.mvpIdx[REF_PIC_LIST_0]] + mvd;
          pu.mv[REF_PIC_LIST_0].mvCliptoStorageBitDepth();
        }
        else
        {
          for ( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            RefPicList eRefList = RefPicList( uiRefListIdx );
            if ((pu.cs->slice->getNumRefIdx(eRefList) > 0 || (eRefList == REF_PIC_LIST_0 && CU::isIBC(*pu.cu))) && (pu.interDir & (1 << uiRefListIdx)))
            {
              AMVPInfo amvpInfo;
              PU::fillMvpCand(pu, eRefList, pu.refIdx[eRefList], amvpInfo);
              pu.mvpNum [eRefList] = amvpInfo.numCand;
              if (!cu.cs->pcv->isEncoder)
              {
                pu.mvd[eRefList].changeTransPrecAmvr2Internal(pu.cu->imv);
              }
              pu.mv[eRefList] = amvpInfo.mvCand[pu.mvpIdx[eRefList]] + pu.mvd[eRefList];
              pu.mv[eRefList].mvCliptoStorageBitDepth();
            }
          }
        }
        PU::spanMotionInfo( pu, mrgCtx );
      }
    }
    if( !cu.geoFlag )
    {
      if( g_mctsDecCheckEnabled && !MCTSHelper::checkMvBufferForMCTSConstraint( pu, true ) )
      {
        printf( "DECODER: pu motion vector across tile boundaries (%d,%d,%d,%d)\n", pu.lx(), pu.ly(), pu.lwidth(), pu.lheight() );
      }
    }
    if (CU::isIBC(cu))
    {
      const int cuPelX = pu.Y().x;
      const int cuPelY = pu.Y().y;
      int roiWidth = pu.lwidth();
      int roiHeight = pu.lheight();
      const unsigned int  lcuWidth = pu.cs->slice->getSPS()->getMaxCUWidth();
      int xPred = pu.mv[0].getHor() >> MV_FRACTIONAL_BITS_INTERNAL;
      int yPred = pu.mv[0].getVer() >> MV_FRACTIONAL_BITS_INTERNAL;
      CHECK_VTM(!m_pcInterPred->isLumaBvValid(lcuWidth, cuPelX, cuPelY, roiWidth, roiHeight, xPred, yPred), "invalid block vector for IBC detected.");
    }
  }
}
//! \}
