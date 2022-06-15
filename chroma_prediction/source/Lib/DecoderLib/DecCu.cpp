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
#if ANNM
#include <torch/script.h>
#include <vector>
#endif
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

void DecCu::decompressCtu(
#if ANNM
  bool &flag0,
#endif
  CodingStructure &cs, const UnitArea &ctuArea)
{
  //====================================================//
#if ANNM
  const int width  = cs.pcv->maxCUWidth;
  const int height = cs.pcv->maxCUHeight;

  bool flag = false;
  Pel *img_CTU_CNN[2];
  for (int k = 0; k < 2; k++)
  {
    img_CTU_CNN[k] = new Pel[int(width / 2) * int(height / 2)];
  }

  // cout << width << "," << height << endl;
  Position pos_luma = ctuArea.lumaPos();
#endif
  //====================================================//
  const int maxNumChannelType = cs.pcv->chrFormat != CHROMA_400 && CS::isDualITree(cs) ? 2 : 1;

  if (cs.resetIBCBuffer)
  {
    m_pcInterPred->resetIBCBuffer(cs.pcv->chrFormat, cs.slice->getSPS()->getMaxCUHeight());
    cs.resetIBCBuffer = false;
  }
  for (int ch = 0; ch < maxNumChannelType; ch++)
  {
    const ChannelType chType = ChannelType(ch);
    Position          prevTmpPos;
    prevTmpPos.x = -1;
    prevTmpPos.y = -1;
#if ANNM
    if (chType == CHANNEL_TYPE_CHROMA)
    {
#if ANNM

      const CodingUnit *cuCurr_L      = cs.getCU(pos_luma, CH_L);
      const CodingUnit *cuLeft_L      = cs.getCU(pos_luma.offset(-width, 0), CH_L);
      const CodingUnit *cuAbove_L     = cs.getCU(pos_luma.offset(0, -height), CH_L);
      const CodingUnit *cuAboveLeft_L = cs.getCU(pos_luma.offset(-width, -height), CH_L);

      Position pos_chroma = ctuArea.chromaPos();

      const CodingUnit *cuCurr_C      = cs.getCU(pos_chroma, CH_C);
      const CodingUnit *cuLeft_C      = cs.getCU(pos_chroma.offset(-int(width / 2), 0), CH_C);
      const CodingUnit *cuAbove_C     = cs.getCU(pos_chroma.offset(0, -int(height / 2)), CH_C);
      const CodingUnit *cuAboveLeft_C = cs.getCU(pos_chroma.offset(-int(width / 2), -int(height / 2)), CH_C);

      if (cuLeft_L != NULL && cuAboveLeft_L != NULL && cuAbove_L != NULL && cuLeft_C != NULL && cuAboveLeft_C != NULL
          && cuAbove_C != NULL)
      {
        flag = true;   //当左边相邻 上边相邻 左上相邻块都可用时，flag为true
      }
      double maxvalue10 = 255.0;
      if (flag && flag0)   //当左边相邻 上边相邻 左上相邻块都可用 并且
                           //这个块没有超过了图片的右边界和下边界，使用ANNM的方法提前重建当前块以供后续使用
      {
        //============================================================================================//
        // pytorch:准备神经网络的输入变量
        //============================================================================================//

        torch::Tensor recluma;   //当前色度块对应的下采样重建亮度块；
        torch::Tensor refrence_y;
        torch::Tensor refrence_u;
        torch::Tensor refrence_v;
        ////////////////////////////////////////////////////
        torch::Tensor refrence_yAbove;   //参考亮度像素的下采样重建值
        torch::Tensor refrence_yAboveLeft;
        torch::Tensor refrence_yLeft;
        torch::Tensor refrence_uAbove;   //参考u分量像素值
        torch::Tensor refrence_uAboveLeft;
        torch::Tensor refrence_uLeft;
        torch::Tensor refrence_vAbove;   //参考v分量像素值
        torch::Tensor refrence_vAboveLeft;
        torch::Tensor refrence_vLeft;
        torch::Tensor refrence_yuv;   //参考YUV分量的合并
                                      ////////////////////////////////////////////////////////
        {                             // Y COMPONETNT
          PelBuf fig0_Above_L     = cs.getRecoBuf(cuAbove_L->block(COMPONENT_Y));
          PelBuf fig1_AboveLeft_L = cs.getRecoBuf(cuAboveLeft_L->block(COMPONENT_Y));
          PelBuf fig2_Left_L      = cs.getRecoBuf(cuLeft_L->block(COMPONENT_Y));
          PelBuf fig3_Curr_L      = cs.getRecoBuf(cuCurr_L->block(COMPONENT_Y));

          std::vector<float> recCurLuma_vec;
          std::vector<float> refenceLuma_Above_vec;
          std::vector<float> refenceLuma_AboveLeft_vec;
          std::vector<float> refenceLuma_Left_vec;
          std::vector<float> refenceCb_Above_vec;
          std::vector<float> refenceCb_AboveLeft_vec;
          std::vector<float> refenceCb_Left_vec;
          std::vector<float> refenceCr_Above_vec;
          std::vector<float> refenceCr_AboveLeft_vec;
          std::vector<float> refenceCr_Left_vec;
          for (int i = 0; i < width; i++)
          {
            if (i % 2 == 0)
            {
              for (int j = 0; j < height; j++)
              {
                if (j % 2 == 0)
                {
                  // fprintf(fp_luma, "%d ", fig3_Curr_L.at(j, i));
                  recCurLuma_vec.push_back(fig3_Curr_L.at(j, i) / maxvalue10);
                }
              }
            }
          }
          for (int i = 0; i < width; i++)
          {
            for (int j = 0; j < height; j++)
            {
              // fprintf(fp_luma, "%d ", fig3_Curr_L.at(j, i));
              // recCurLuma_vec.push_back(fig3_Curr_L.at(i, j) / 255.0);
            }
          }
          // fclose(fp_luma);
          recluma = torch::tensor(recCurLuma_vec);
          recluma = recluma.view({ 1, 1, 64, 64 });
#if 1
          //相邻块正上方的亮度参考
          for (int i = 120; i < 128; i++)   //高
          {
            if (i % 2 == 0)
            {
              for (int j = 0; j < 128; j++)   //宽
              {
                if (j % 2 == 0)
                {
                  // fprintf(fp_luma, "%d ", fig3_Curr_L.at(j, i));
                  refenceLuma_Above_vec.push_back(fig0_Above_L.at(j, i) / maxvalue10);
                }
              }
            }
          }
#endif
#if 0
      for (int i = 120; i < 128; i++)
      {
        
        
          for (int j = 0; j < 128; j++)
          {
           
            
             fprintf(fp_luma, "%d ", fig0_Above_L.at(j, i));
              //refenceLuma_Above_vec.push_back(fig0_Above_L.at(j, i) );
            
          }
        
      }
      fclose(fp_luma);
#endif

          refrence_yAbove = torch::tensor(refenceLuma_Above_vec);
          refrence_yAbove = refrence_yAbove.view({ 4, 64 });
          refrence_yAbove = refrence_yAbove.t();
// refrence_yAbove = refrence_yAbove.numpy_T();
#if 0
      for (int i = 0; i < 128; i++)
      {
        for (int j = 0; j < 8; j++)
        {
          fprintf(fp_luma, "%d ", refrence_yAbove[i][j]);
        }
      }
      fclose(fp_luma);
#endif
          refrence_yAbove = refrence_yAbove.view({ 1, 1, 64, 4 });   //转置之后的正上方参考像素
          //相邻块左上方的亮度参考
          for (int i = 120; i < 128; i++)
          {
            if (i % 2 == 0)
            {
              for (int j = 120; j < 128; j++)
              {
                if (j % 2 == 0)
                {
                  refenceLuma_AboveLeft_vec.push_back(fig1_AboveLeft_L.at(j, i) / maxvalue10);
                }
              }
            }
          }
          refrence_yAboveLeft = torch::tensor(refenceLuma_AboveLeft_vec);

          refrence_yAboveLeft = refrence_yAboveLeft.view({ 1, 1, 4, 4 });
          //相邻左侧的亮度参考像素
          for (int i = 0; i < 128; i++)
          {
            if (i % 2 == 0)
            {
              for (int j = 120; j < 128; j++)
              {
                if (j % 2 == 0)
                {
                  refenceLuma_Left_vec.push_back(fig2_Left_L.at(j, i) / maxvalue10);
                }
              }
            }
          }
          refrence_yLeft = torch::tensor(refenceLuma_Left_vec);
          refrence_yLeft = refrence_yLeft.view({ 1, 1, 64, 4 });
          refrence_y     = torch::cat({ refrence_yAbove, refrence_yAboveLeft, refrence_yLeft }, 2);
        }

        {   // U COMPONETNT
          PelBuf fig0_Above_L     = cs.getRecoBuf(cuAbove_C->block(COMPONENT_Cb));
          PelBuf fig1_AboveLeft_L = cs.getRecoBuf(cuAboveLeft_C->block(COMPONENT_Cb));
          PelBuf fig2_Left_L      = cs.getRecoBuf(cuLeft_C->block(COMPONENT_Cb));

          std::vector<float> refenceCb_Above_vec;
          std::vector<float> refenceCb_AboveLeft_vec;
          std::vector<float> refenceCb_Left_vec;

          for (int i = 0; i < 64; i++)
          {
            for (int j = 60; j < 64; j++)
            {
              refenceCb_Above_vec.push_back(fig0_Above_L.at(j, i) / 255.0);
            }
          }
          refrence_uAbove = torch::tensor(refenceCb_Above_vec);

          refrence_uAbove = refrence_uAbove.view({ 4, 64 });
          refrence_uAbove = refrence_uAbove.t();
          refrence_uAbove = refrence_uAbove.view({ 1, 1, 64, 4 });   //转置之后的正上方参考像素

          for (int i = 60; i < 64; i++)
          {
            for (int j = 60; j < 64; j++)
            {
              refenceCb_AboveLeft_vec.push_back(fig1_AboveLeft_L.at(j, i) / maxvalue10);
            }
          }

          refrence_uAboveLeft = torch::tensor(refenceCb_AboveLeft_vec);
          refrence_uAboveLeft = refrence_uAboveLeft.view({ 1, 1, 4, 4 });

          for (int i = 0; i < 64; i++)
          {
            for (int j = 60; j < 64; j++)
            {
              refenceCb_Left_vec.push_back(fig2_Left_L.at(j, i) / maxvalue10);
            }
          }

          refrence_uLeft = torch::tensor(refenceCb_Left_vec);
          refrence_uLeft = refrence_uLeft.view({ 1, 1, 64, 4 });
          refrence_u     = torch::cat({ refrence_uAbove, refrence_uAboveLeft, refrence_uLeft }, 2);
        }

        {   // V COMPONETNT
          PelBuf fig0_Above_L     = cs.getRecoBuf(cuAbove_C->block(COMPONENT_Cr));
          PelBuf fig1_AboveLeft_L = cs.getRecoBuf(cuAboveLeft_C->block(COMPONENT_Cr));
          PelBuf fig2_Left_L      = cs.getRecoBuf(cuLeft_C->block(COMPONENT_Cr));

          std::vector<float> refenceCr_Above_vec;
          std::vector<float> refenceCr_AboveLeft_vec;
          std::vector<float> refenceCr_Left_vec;

          for (int i = 0; i < 64; i++)
          {
            for (int j = 60; j < 64; j++)
            {
              refenceCr_Above_vec.push_back(fig0_Above_L.at(j, i) / maxvalue10);
            }
          }
          refrence_vAbove = torch::tensor(refenceCr_Above_vec);
          refrence_vAbove = refrence_vAbove.view({ 4, 64 });
          refrence_vAbove = refrence_vAbove.t();
          refrence_vAbove = refrence_vAbove.view({ 1, 1, 64, 4 });   //转置之后的正上方参考像素

          for (int i = 60; i < 64; i++)
          {
            for (int j = 60; j < 64; j++)
            {
              refenceCr_AboveLeft_vec.push_back(fig1_AboveLeft_L.at(j, i) / maxvalue10);
            }
          }

          refrence_vAboveLeft = torch::tensor(refenceCr_AboveLeft_vec);
          refrence_vAboveLeft = refrence_vAboveLeft.view({ 1, 1, 4, 4 });

          for (int i = 0; i < 64; i++)
          {
            for (int j = 60; j < 64; j++)
            {
              refenceCr_Left_vec.push_back(fig2_Left_L.at(j, i) / maxvalue10);
            }
          }

          refrence_vLeft = torch::tensor(refenceCr_Left_vec);
          refrence_vLeft = refrence_vLeft.view({ 1, 1, 64, 4 });
          refrence_v     = torch::cat({ refrence_vAbove, refrence_vAboveLeft, refrence_vLeft }, 2);
        }

        // Do the prediction
        refrence_yuv = torch::cat({ refrence_y, refrence_u, refrence_v }, 1);   //注意语法，要合并的tensor使用{}
        torch::jit::script::Module module;
        module = torch::jit::load("G:/思路/ToPt/128x128_49th.pt");

        at::Tensor output      = module.forward({ recluma, refrence_yuv }).toTensor();
        at::Tensor True_output = output * maxvalue10;
        for (int k = 0; k < 2; k++)
        {
          for (int uiY = 0; uiY < int(width / 2); uiY++)
          {
            for (int uiX = 0; uiX < int(height / 2); uiX++)
            {
              if (k == 0)
              {
                short a = True_output[0][k][uiY][uiX].item().toShort();
                // fprintf(fp_Cb, "%d ", a);
              }

              else
              {
                short b = True_output[0][k][uiY][uiX].item().toShort();
                // fprintf(fp_Cr, "%d ",b );
              }
              img_CTU_CNN[k][uiY * int(height / 2) + uiX] = True_output[0][k][uiY][uiX].item().toShort();
            }
          }
        }
      }
      else
      {
        for (int k = 0; k < 2; k++)
        {
          for (int uiY = 0; uiY < int(width / 2); uiY++)
          {
            for (int uiX = 0; uiX < int(height / 2); uiX++)
            {
              img_CTU_CNN[k][uiY * int(height / 2) + uiX] = maxvalue10/2;
            }
          }
        }
      }
    }
#endif
    //====================================================//

#endif
    for (auto &currCU: cs.traverseCUs(CS::getArea(cs, ctuArea, chType), chType))
    {
      if (currCU.Y().valid())
      {
        const int vSize = cs.slice->getSPS()->getMaxCUHeight() > 64 ? 64 : cs.slice->getSPS()->getMaxCUHeight();
        if ((currCU.Y().x % vSize) == 0 && (currCU.Y().y % vSize) == 0)
        {
          for (int x = currCU.Y().x; x < currCU.Y().x + currCU.Y().width; x += vSize)
          {
            for (int y = currCU.Y().y; y < currCU.Y().y + currCU.Y().height; y += vSize)
            {
              m_pcInterPred->resetVPDUforIBC(cs.pcv->chrFormat, cs.slice->getSPS()->getMaxCUHeight(), vSize,
                                             x + g_IBCBufferSize / cs.slice->getSPS()->getMaxCUHeight() / 2, y);
            }
          }
        }
      }
      if (currCU.predMode != MODE_INTRA && currCU.predMode != MODE_PLT && currCU.Y().valid())
      {
        xDeriveCUMV(currCU);

#if K0149_BLOCK_STATISTICS
        if (currCU.geoFlag)
        {
          storeGeoMergeCtx(m_geoMrgCtx);
        }
#endif
      }
#if ANNM
      const int uiLPelX = currCU.chromaPos().x;
      const int uiTPelY = currCU.chromaPos().y;
#endif
      switch (currCU.predMode)
      {
      case MODE_INTER:
      case MODE_IBC:
        xReconInter(
#if ANNM
          img_CTU_CNN, uiLPelX, uiTPelY,
#endif
          currCU);
        break;
      case MODE_PLT:
      case MODE_INTRA:
        xReconIntraQT(
#if ANNM
          img_CTU_CNN, uiLPelX, uiTPelY,
#endif
          currCU);
        break;
      default: THROW("Invalid prediction mode"); break;
      }

      m_pcInterPred->xFillIBCBuffer(currCU);

      DTRACE_BLOCK_REC(cs.picture->getRecoBuf(currCU), currCU, currCU.predMode);
    }
  }
}
#if K0149_BLOCK_STATISTICS
  getAndStoreBlockStatistics(cs, ctuArea);
#endif


// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

void DecCu::xIntraRecBlk( 
#if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  TransformUnit &tu, const ComponentID compID)
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
          m_pcIntraPred->predIntraAng(
#if ANNM
      img, pieX, pieY,
#endif
            compID, piPredReg, pu);
        }
      }
      else
        m_pcIntraPred->predIntraAng(
#if ANNM
      img, pieX, pieY,
#endif
          compID, piPred, pu);
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

void DecCu::xIntraRecACTBlk(
#if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  TransformUnit &tu)
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
      m_pcIntraPred->predIntraAng(
#if ANNM
      img, pieX, pieY,

#endif
        compID, piPred, pu);
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

void DecCu::xReconIntraQT( 
  #if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  CodingUnit &cu)
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
    xIntraRecACTQT(
#if ANNM
        img, pieX, pieY,
#endif
      cu);
  }
  else
  {
  const uint32_t numChType = ::getNumberValidChannels( cu.chromaFormat );

  for( uint32_t chType = CHANNEL_TYPE_LUMA; chType < numChType; chType++ )
  {
    if( cu.blocks[chType].valid() )
    {
      xIntraRecQT( 
#if ANNM
        img, pieX, pieY,
#endif
        cu, ChannelType(chType));
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
DecCu::xIntraRecQT(
#if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  CodingUnit &cu, const ChannelType chType)
{
  for( auto &currTU : CU::traverseTUs( cu ) )
  {
    if( isLuma( chType ) )
    {
      xIntraRecBlk(
#if ANNM
        img, pieX, pieY,
#endif
        currTU, COMPONENT_Y);
    }
    else
    {
      const uint32_t numValidComp = getNumberValidComponents( cu.chromaFormat );

      for( uint32_t compID = COMPONENT_Cb; compID < numValidComp; compID++ )
      {
        xIntraRecBlk(
#if ANNM
          img, pieX, pieY,
#endif
          currTU, ComponentID(compID));
      }
    }
  }
}

void DecCu::xIntraRecACTQT(
#if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  CodingUnit &cu)
{
  for (auto &currTU : CU::traverseTUs(cu))
  {
    xIntraRecACTBlk(
#if ANNM
        img, pieX, pieY,
#endif
      currTU);
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

void DecCu::xReconInter(
#if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  CodingUnit &cu)
{
  if( cu.geoFlag )
  {
    m_pcInterPred->motionCompensationGeo( cu, m_geoMrgCtx );
    PU::spanGeoMotionInfo( *cu.firstPU, m_geoMrgCtx, cu.firstPU->geoSplitDir, cu.firstPU->geoMergeIdx0, cu.firstPU->geoMergeIdx1 );
  }
  else
  {
  m_pcIntraPred->geneIntrainterPred(
#if ANNM
    img, pieX, pieY,
#endif
    cu);

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
