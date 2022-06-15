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

/** \file     EncCu.cpp
    \brief    Coding Unit (CU) encoder class
*/

#include "EncCu.h"

#include "EncLib.h"
#include "Analyze.h"
#include "AQp.h"

#include "CommonLib/dtrace_codingstruct.h"
#include "CommonLib/Picture.h"
#include "CommonLib/UnitTools.h"
#include "MCTS.h"


#include "CommonLib/dtrace_buffer.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>





#if ANNM
#include <torch/script.h>
#include <vector>
#endif
#if 0
#include <stdio.h>
FILE *fp_lumacur   = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/lumacur.txt", "a");
FILE *fp_luma    = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/luma.txt", "a");
FILE *fp_Cb     = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/Cb.txt", "a");
FILE *fp_Cr = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/Cr.txt", "a");
FILE *fp_Cb1 = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/Cbpre.txt", "a");
FILE *fp_Cr1 = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/Crpre.txt", "a");
//FILE *lumaround = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/lumaround.txt", "a");
//FILE *cbround   = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/cbround.txt", "a");
//FILE *crround   = fopen("F:/VTM8.2参考LINWEIZHU/VVCSoftware_VTM-VTM-8.2/bin/vs16/msvc-19.26/x86_64/debug/crround.txt", "a");
#endif
//! \ingroup EncoderLib
//! \{

// ====================================================================================================================
EncCu::EncCu() : m_GeoModeTest
{
  GeoMotionInfo(0, 1), GeoMotionInfo(1, 0),GeoMotionInfo(0, 2), GeoMotionInfo(1, 2), GeoMotionInfo(2, 0),
  GeoMotionInfo(2, 1), GeoMotionInfo(0, 3),GeoMotionInfo(1, 3), GeoMotionInfo(2, 3), GeoMotionInfo(3, 0),
  GeoMotionInfo(3, 1), GeoMotionInfo(3, 2),GeoMotionInfo(0, 4), GeoMotionInfo(1, 4), GeoMotionInfo(2, 4),
  GeoMotionInfo(3, 4), GeoMotionInfo(4, 0),GeoMotionInfo(4, 1), GeoMotionInfo(4, 2), GeoMotionInfo(4, 3),
  GeoMotionInfo(0, 5), GeoMotionInfo(1, 5),GeoMotionInfo(2, 5), GeoMotionInfo(3, 5), GeoMotionInfo(4, 5),
  GeoMotionInfo(5, 0), GeoMotionInfo(5, 1),GeoMotionInfo(5, 2), GeoMotionInfo(5, 3), GeoMotionInfo(5, 4)
}
{}

void EncCu::create( EncCfg* encCfg )
{
  unsigned      uiMaxWidth    = encCfg->getMaxCUWidth();
  unsigned      uiMaxHeight   = encCfg->getMaxCUHeight();
  ChromaFormat  chromaFormat  = encCfg->getChromaFormatIdc();

  unsigned      numWidths     = gp_sizeIdxInfo->numWidths();
  unsigned      numHeights    = gp_sizeIdxInfo->numHeights();
  m_pTempCS = new CodingStructure**  [numWidths];
  m_pBestCS = new CodingStructure**  [numWidths];
  m_pTempCS2 = new CodingStructure** [numWidths];
  m_pBestCS2 = new CodingStructure** [numWidths];

  for( unsigned w = 0; w < numWidths; w++ )
  {
    m_pTempCS[w] = new CodingStructure*  [numHeights];
    m_pBestCS[w] = new CodingStructure*  [numHeights];
    m_pTempCS2[w] = new CodingStructure* [numHeights];
    m_pBestCS2[w] = new CodingStructure* [numHeights];

    for( unsigned h = 0; h < numHeights; h++ )
    {
      unsigned width  = gp_sizeIdxInfo->sizeFrom( w );
      unsigned height = gp_sizeIdxInfo->sizeFrom( h );

      if( gp_sizeIdxInfo->isCuSize( width ) && gp_sizeIdxInfo->isCuSize( height ) )
      {
        m_pTempCS[w][h] = new CodingStructure( m_unitCache.cuCache, m_unitCache.puCache, m_unitCache.tuCache );
        m_pBestCS[w][h] = new CodingStructure( m_unitCache.cuCache, m_unitCache.puCache, m_unitCache.tuCache );

        m_pTempCS[w][h]->create(chromaFormat, Area(0, 0, width, height), false, (bool)encCfg->getPLTMode());
        m_pBestCS[w][h]->create(chromaFormat, Area(0, 0, width, height), false, (bool)encCfg->getPLTMode());

        m_pTempCS2[w][h] = new CodingStructure( m_unitCache.cuCache, m_unitCache.puCache, m_unitCache.tuCache );
        m_pBestCS2[w][h] = new CodingStructure( m_unitCache.cuCache, m_unitCache.puCache, m_unitCache.tuCache );

        m_pTempCS2[w][h]->create(chromaFormat, Area(0, 0, width, height), false, (bool)encCfg->getPLTMode());
        m_pBestCS2[w][h]->create(chromaFormat, Area(0, 0, width, height), false, (bool)encCfg->getPLTMode());
      }
      else
      {
        m_pTempCS[w][h] = nullptr;
        m_pBestCS[w][h] = nullptr;
        m_pTempCS2[w][h] = nullptr;
        m_pBestCS2[w][h] = nullptr;
      }
    }
  }

  m_cuChromaQpOffsetIdxPlus1 = 0;

  unsigned maxDepth = numWidths + numHeights;

  m_modeCtrl = new EncModeCtrlMTnoRQT();

  m_modeCtrl->create( *encCfg );

  for (unsigned ui = 0; ui < MMVD_MRG_MAX_RD_BUF_NUM; ui++)
  {
    m_acMergeBuffer[ui].create( chromaFormat, Area( 0, 0, uiMaxWidth, uiMaxHeight ) );
  }
  for (unsigned ui = 0; ui < MRG_MAX_NUM_CANDS; ui++)
  {
    m_acRealMergeBuffer[ui].create(chromaFormat, Area(0, 0, uiMaxWidth, uiMaxHeight));
    m_acMergeTmpBuffer[ui].create(chromaFormat, Area(0, 0, uiMaxWidth, uiMaxHeight));
  }
  for( unsigned ui = 0; ui < GEO_MAX_TRY_WEIGHTED_SAD; ui++ )
  {
    m_acGeoWeightedBuffer[ui].create( chromaFormat, Area( 0, 0, uiMaxWidth, uiMaxHeight ) );
  }

  m_CtxBuffer.resize( maxDepth );
  m_CurrCtx = 0;
}


void EncCu::destroy()
{
  unsigned numWidths  = gp_sizeIdxInfo->numWidths();
  unsigned numHeights = gp_sizeIdxInfo->numHeights();

  for( unsigned w = 0; w < numWidths; w++ )
  {
    for( unsigned h = 0; h < numHeights; h++ )
    {
      if( m_pBestCS[w][h] ) m_pBestCS[w][h]->destroy();
      if( m_pTempCS[w][h] ) m_pTempCS[w][h]->destroy();

      delete m_pBestCS[w][h];
      delete m_pTempCS[w][h];

      if( m_pBestCS2[w][h] ) m_pBestCS2[w][h]->destroy();
      if( m_pTempCS2[w][h] ) m_pTempCS2[w][h]->destroy();

      delete m_pBestCS2[w][h];
      delete m_pTempCS2[w][h];
    }

    delete[] m_pTempCS[w];
    delete[] m_pBestCS[w];
    delete[] m_pTempCS2[w];
    delete[] m_pBestCS2[w];
  }

  delete[] m_pBestCS; m_pBestCS = nullptr;
  delete[] m_pTempCS; m_pTempCS = nullptr;
  delete[] m_pBestCS2; m_pBestCS2 = nullptr;
  delete[] m_pTempCS2; m_pTempCS2 = nullptr;

#if REUSE_CU_RESULTS
  if (m_tmpStorageLCU)
  {
    m_tmpStorageLCU->destroy();
    delete m_tmpStorageLCU;  m_tmpStorageLCU = nullptr;
  }
#endif

#if REUSE_CU_RESULTS
  m_modeCtrl->destroy();

#endif
  delete m_modeCtrl;
  m_modeCtrl = nullptr;

  for (unsigned ui = 0; ui < MMVD_MRG_MAX_RD_BUF_NUM; ui++)
  {
    m_acMergeBuffer[ui].destroy();
  }
  for (unsigned ui = 0; ui < MRG_MAX_NUM_CANDS; ui++)
  {
    m_acRealMergeBuffer[ui].destroy();
    m_acMergeTmpBuffer[ui].destroy();
  }
  for (unsigned ui = 0; ui < GEO_MAX_TRY_WEIGHTED_SAD; ui++)
  {
    m_acGeoWeightedBuffer[ui].destroy();
  }
}



EncCu::~EncCu()
{
}



/** \param    pcEncLib      pointer of encoder class
 */
void EncCu::init( EncLib* pcEncLib, const SPS& sps PARL_PARAM( const int tId ) )
{
  m_pcEncCfg           = pcEncLib;
  m_pcIntraSearch      = pcEncLib->getIntraSearch( PARL_PARAM0( tId ) );
  m_pcInterSearch      = pcEncLib->getInterSearch( PARL_PARAM0( tId ) );
  m_pcTrQuant          = pcEncLib->getTrQuant( PARL_PARAM0( tId ) );
  m_pcRdCost           = pcEncLib->getRdCost ( PARL_PARAM0( tId ) );
  m_CABACEstimator     = pcEncLib->getCABACEncoder( PARL_PARAM0( tId ) )->getCABACEstimator( &sps );
  m_CABACEstimator->setEncCu(this);
  m_CtxCache           = pcEncLib->getCtxCache( PARL_PARAM0( tId ) );
  m_pcRateCtrl         = pcEncLib->getRateCtrl();
  m_pcSliceEncoder     = pcEncLib->getSliceEncoder();
#if ENABLE_SPLIT_PARALLELISM
  m_pcEncLib           = pcEncLib;
  m_dataId             = tId;
#endif
  m_pcLoopFilter       = pcEncLib->getLoopFilter();
  m_GeoCostList.init(GEO_NUM_PARTITION_MODE, m_pcEncCfg->getMaxNumGeoCand());
  m_AFFBestSATDCost = MAX_DOUBLE;

  DecCu::init( m_pcTrQuant, m_pcIntraSearch, m_pcInterSearch );

  m_modeCtrl->init( m_pcEncCfg, m_pcRateCtrl, m_pcRdCost );

  m_pcInterSearch->setModeCtrl( m_modeCtrl );
  m_modeCtrl->setInterSearch(m_pcInterSearch);
  m_pcIntraSearch->setModeCtrl( m_modeCtrl );

}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

void EncCu::compressCtu( CodingStructure& cs, const UnitArea& area, const unsigned ctuRsAddr, const int prevQP[], const int currQP[] )
{
  m_modeCtrl->initCTUEncoding( *cs.slice );
  cs.treeType = TREE_D;

  cs.slice->m_mapPltCost[0].clear();
  cs.slice->m_mapPltCost[1].clear();
#if ENABLE_SPLIT_PARALLELISM
  if( m_pcEncCfg->getNumSplitThreads() > 1 )
  {
    for( int jId = 1; jId < NUM_RESERVERD_SPLIT_JOBS; jId++ )
    {
      EncCu*            jobEncCu  = m_pcEncLib->getCuEncoder( cs.picture->scheduler.getSplitDataId( jId ) );
      CacheBlkInfoCtrl* cacheCtrl = dynamic_cast< CacheBlkInfoCtrl* >( jobEncCu->m_modeCtrl );
#if REUSE_CU_RESULTS
      BestEncInfoCache* bestCache = dynamic_cast< BestEncInfoCache* >( jobEncCu->m_modeCtrl );
#endif
      SaveLoadEncInfoSbt *sbtCache = dynamic_cast< SaveLoadEncInfoSbt* >( jobEncCu->m_modeCtrl );
      if( cacheCtrl )
      {
        cacheCtrl->init( *cs.slice );
      }
#if REUSE_CU_RESULTS
      if (bestCache)
      {
        bestCache->init(*cs.slice);
      }
#endif
      if (sbtCache)
      {
        sbtCache->init(*cs.slice);
      }
    }
  }

#if REUSE_CU_RESULTS
  if( auto* cacheCtrl = dynamic_cast<BestEncInfoCache*>( m_modeCtrl ) ) { cacheCtrl->tick(); }
#endif
  if( auto* cacheCtrl = dynamic_cast<CacheBlkInfoCtrl*>( m_modeCtrl ) ) { cacheCtrl->tick(); }
#endif
  // init the partitioning manager
  QTBTPartitioner partitioner;
  partitioner.initCtu(area, CH_L, *cs.slice);
  if (m_pcEncCfg->getIBCMode())
  {
    if (area.lx() == 0 && area.ly() == 0)
    {
      m_pcInterSearch->resetIbcSearch();
    }
    m_pcInterSearch->resetCtuRecord();
    m_ctuIbcSearchRangeX = m_pcEncCfg->getIBCLocalSearchRangeX();
    m_ctuIbcSearchRangeY = m_pcEncCfg->getIBCLocalSearchRangeY();
  }
  if (m_pcEncCfg->getIBCMode() && m_pcEncCfg->getIBCHashSearch() && (m_pcEncCfg->getIBCFastMethod() & IBC_FAST_METHOD_ADAPTIVE_SEARCHRANGE))
  {
    const int hashHitRatio = m_ibcHashMap.getHashHitRatio(area.Y()); // in percent
    if (hashHitRatio < 5) // 5%
    {
      m_ctuIbcSearchRangeX >>= 1;
      m_ctuIbcSearchRangeY >>= 1;
    }
    if (cs.slice->getNumRefIdx(REF_PIC_LIST_0) > 0)
    {
      m_ctuIbcSearchRangeX >>= 1;
      m_ctuIbcSearchRangeY >>= 1;
    }
  }
  // init current context pointer
  m_CurrCtx = m_CtxBuffer.data();

  CodingStructure *tempCS = m_pTempCS[gp_sizeIdxInfo->idxFrom( area.lumaSize().width )][gp_sizeIdxInfo->idxFrom( area.lumaSize().height )];
  CodingStructure *bestCS = m_pBestCS[gp_sizeIdxInfo->idxFrom( area.lumaSize().width )][gp_sizeIdxInfo->idxFrom( area.lumaSize().height )];

  cs.initSubStructure(*tempCS, partitioner.chType, partitioner.currArea(), false);
  cs.initSubStructure(*bestCS, partitioner.chType, partitioner.currArea(), false);
  tempCS->currQP[CH_L] = bestCS->currQP[CH_L] =
  tempCS->baseQP       = bestCS->baseQP       = currQP[CH_L];
  tempCS->prevQP[CH_L] = bestCS->prevQP[CH_L] = prevQP[CH_L];
#if ANNM
  const int width         = bestCS->pcv->maxCUWidth;
  const int height        = bestCS->pcv->maxCUHeight;

  bool flag  = false;
  bool flag1 = true;
  Pel *img_CTU_CNN[2];
  for (int k = 0; k < 2; k++)
  {
    img_CTU_CNN[k] = new Pel[int(width / 2) * int(height / 2)];   
  }
#endif
  xCompressCU(
#if ANNM
    flag, img_CTU_CNN,
#endif
    tempCS, bestCS,partitioner);
#if PRINT
  // printf("(%d,%d) %d %d: sum bits is %lld\n", bestCS->area.blocks[0].x, bestCS->area.blocks[0].y,
  // bestCS->area.blocks[0].width, bestCS->area.blocks[0].height, bestCS->fracBits);
  static uint64_t sumluma = 0;
  sumluma += (bestCS->fracBits >> 15);
  printf("(%d,%d) %d %d: luma bits is %lld\n", bestCS->area.blocks[0].x, bestCS->area.blocks[0].y,
         bestCS->area.blocks[0].width, bestCS->area.blocks[0].height, sumluma);
#endif
  cs.slice->m_mapPltCost[0].clear();
  cs.slice->m_mapPltCost[1].clear();
  // all signals were already copied during compression if the CTU was split - at this point only the structures are copied to the top level CS
  const bool copyUnsplitCTUSignals = bestCS->cus.size() == 1;
  cs.useSubStructure(*bestCS, partitioner.chType, CS::getArea(*bestCS, area, partitioner.chType), copyUnsplitCTUSignals,
                     false, false, copyUnsplitCTUSignals);
   //====================================================//


#if ANNM
  Position pos_L = area.lumaPos();

  const CodingUnit* cuCurr_L = bestCS->getCU(pos_L, CH_L);
  const CodingUnit* cuLeft_L = bestCS->getCU(pos_L.offset(-width, 0), CH_L);
  const CodingUnit* cuAbove_L = bestCS->getCU(pos_L.offset(0, -height), CH_L);
  const CodingUnit* cuAboveLeft_L = bestCS->getCU(pos_L.offset(-width, -height), CH_L);

  Position pos_C = area.chromaPos();
  //const CodingUnit *cuCurr_C1     = cs.getOrgBuf(area.block[COMPONENT_Cb]);
 
  const CodingUnit* cuLeft_C = bestCS->getCU(pos_C.offset(-int(width/2), 0), CH_C);
  const CodingUnit* cuAbove_C = bestCS->getCU(pos_C.offset(0, -int(height/2)), CH_C);
  const CodingUnit* cuAboveLeft_C = bestCS->getCU(pos_C.offset(-int(width/2), -int(height/2)), CH_C);

  if (cuLeft_L != NULL && cuAboveLeft_L != NULL && cuAbove_L != NULL && cuLeft_C != NULL && cuAboveLeft_C != NULL && cuAbove_C != NULL)
  {
    flag = true;//当左边相邻 上边相邻 左上相邻块都可用时，flag为true
  }

  const int pic_width = bestCS->pcv->lumaWidth;
  const int pic_height = bestCS->pcv->lumaHeight; 

  //cout << pic_width << "," << pic_height << endl;

  const int uiLPelX = bestCS->area.Y().lumaPos().x;
  const int uiRPelX = uiLPelX + width - 1;
  const int uiTPelY = bestCS->area.Y().lumaPos().y;
  const int uiBPelY = uiTPelY + height - 1;

  if (uiRPelX >= pic_width || uiBPelY >= pic_height)
  {
    flag  = false;//判断这个块是否超过了图片的右边界和下边界，若超过，则flag为false
    flag1 = false;
  }
  double maxvalue10 = 255.0;
  //double maxvalue8  = 255.0;
  if (flag)//当左边相邻 上边相邻 左上相邻块都可用 并且  这个块没有超过了图片的右边界和下边界，使用ANNM的方法提前重建当前块以供后续使用
  {
    
    //============================================================================================//
    //pytorch:准备神经网络的输入变量
    //============================================================================================//
    
    torch::Tensor recluma;        //当前色度块对应的下采样重建亮度块；
    torch::Tensor refrence_y;
    torch::Tensor refrence_u;
    torch::Tensor refrence_v; 
    ////////////////////////////////////////////////////
    torch::Tensor refrence_yAbove;     //参考亮度像素的下采样重建值
    torch::Tensor refrence_yAboveLeft;
    torch::Tensor refrence_yLeft;
    torch::Tensor refrence_uAbove;     //参考u分量像素值
    torch::Tensor refrence_uAboveLeft;
    torch::Tensor refrence_uLeft;
    torch::Tensor refrence_vAbove;   //参考v分量像素值
    torch::Tensor refrence_vAboveLeft;
    torch::Tensor refrence_vLeft;
    torch::Tensor refrence_yuv;   //参考YUV分量的合并
   
    
   ////////////////////////////////////////////////////////
    { //Y COMPONETNT
      PelBuf fig0_Above_L       = cs.getRecoBuf(cuAbove_L->block(COMPONENT_Y));
      PelBuf fig1_AboveLeft_L = cs.getRecoBuf(cuAboveLeft_L->block(COMPONENT_Y));
      PelBuf fig2_Left_L          = cs.getRecoBuf(cuLeft_L->block(COMPONENT_Y));
      PelBuf fig3_Curr_L            = cs.getRecoBuf(cuCurr_L->block(COMPONENT_Y));
    

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
      #if YANZHENG
      std::vector<float> refenceLuma_Above_vec_YANZHENG;
      std::vector<float> refenceCb_Above_vec_YANZHENG;
      std::vector<float> refenceCr_Above_vec_YANZHENG;
      std::vector<float> refenceLuma_AboveLeft_vec_YANZHENG;
      std::vector<float> refenceLuma_Left_vec_YANZHENG;
      torch::Tensor      refrence_yAbove_YANZHENG;
      torch::Tensor      refrence_yAboveLeft_YANZHENG;
     // torch::Tensor      refenceLuma_Left_YANZHENG;
      torch::Tensor      refrence_uAbove_YANZHENG;
      torch::Tensor      refrence_vAbove_YANZHENG;
      torch::Tensor      refrence_yLeft_YANZHENG;
      torch::Tensor      refrence_y_YANZHENG;
      #endif
      for (int i = 0; i < width; i++)
      {
        if (i % 2 == 0)
        {
          for (int j = 0; j < height; j++)
          {
            if (j % 2 == 0)
            {
              
              recCurLuma_vec.push_back(fig3_Curr_L.at(j, i) / maxvalue10);
              
              
            }
          }
        }
#if YANZHENG1
        
#endif
      }
      //fclose(fp_lumacur);
      for (int i = 0; i < width; i++)
      {
      
       
          for (int j = 0; j < height; j++)
          {
            
            //fprintf(fp_luma, "%d ", fig3_Curr_L.at(j, i));
              //recCurLuma_vec.push_back(fig3_Curr_L.at(i, j) / 255.0);
              
          }
        
      }
      //fclose(fp_luma);
      recluma = torch::tensor(recCurLuma_vec);
      recluma = recluma.view({ 1, 1, 64, 64});
     #if 1
      //相邻块正上方的亮度参考
      for (int i = 120; i < 128; i++)//高
      {
        if (i%2==0) 
        {
          for (int j = 0; j < 128; j++)//宽
          {
            if (j % 2 == 0) 
            {
              //fprintf(fp_luma, "%d ", fig3_Curr_L.at(j, i));
              refenceLuma_Above_vec.push_back(fig0_Above_L.at(j, i) / maxvalue10);
            }
            
          }
        }
        
      }
      #endif
      #if YANZHENG
      for (int i = 0; i < 128; i++)
      {
        
        
          for (int j = 0; j < 128; j++)
          {
           
            
             
             refenceLuma_Above_vec_YANZHENG.push_back(fig0_Above_L.at(j, i) );
            
          }
        
      }
      refrence_yAbove_YANZHENG = torch::tensor(refenceLuma_Above_vec_YANZHENG);
      refrence_yAbove_YANZHENG = refrence_yAbove_YANZHENG.view({ 128, 128 });
      refrence_yAbove_YANZHENG = refrence_yAbove_YANZHENG.t();
      refrence_yAbove_YANZHENG = refrence_yAbove_YANZHENG.view({ 1, 1, 128, 128 });
      //相邻块左上方的亮度参考
      for (int i = 0; i < 128; i++)
      {
        
        
          for (int j = 0; j < 128; j++)
          {
           
            
              refenceLuma_AboveLeft_vec_YANZHENG.push_back(fig1_AboveLeft_L.at(j, i));
            
          }
        
      }
      refrence_yAboveLeft_YANZHENG = torch::tensor(refenceLuma_AboveLeft_vec_YANZHENG);

      refrence_yAboveLeft_YANZHENG = refrence_yAboveLeft_YANZHENG.view({ 1, 1, 128, 128 });
      //相邻左侧的亮度参考像素
      for (int i = 0; i < 128; i++)
      {
 
        
          for (int j = 0; j < 128; j++)
          {
            
            
              refenceLuma_Left_vec_YANZHENG.push_back(fig2_Left_L.at(j, i));
            
          }
        
      }
      refrence_yLeft_YANZHENG   = torch::tensor(refenceLuma_Left_vec_YANZHENG);
      refrence_yLeft_YANZHENG = refrence_yLeft_YANZHENG.view({ 1, 1, 128, 128 });
      refrence_y_YANZHENG =torch::cat({ refrence_yAbove_YANZHENG, refrence_yAboveLeft_YANZHENG, refrence_yLeft_YANZHENG }, 2); 
      for (int i = 0; i < 384; i++)
      {
        for (int j = 0; j < 128; j++)
        {
          fprintf(fp_luma, "%d ", refrence_y_YANZHENG[0][0][i][j].item().toShort());
        }
      }
      fclose(fp_luma);
      #endif
   
      
      refrence_yAbove = torch::tensor(refenceLuma_Above_vec);
      refrence_yAbove = refrence_yAbove.view({  4, 64 });
      refrence_yAbove = refrence_yAbove.t();
      
     #if YANZHENG1
      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 4; j++)
        {
          fprintf(fp_luma, "%f ", refrence_yAbove[i][j].item().toFloat());
        }
      }
      //fclose(fp_luma);
     #endif
      refrence_yAbove = refrence_yAbove.view({ 1, 1, 64, 4 });//转置之后的正上方参考像素
      //相邻块左上方的亮度参考
      for (int i = 120; i < 128; i++)
      {
        if (i % 2 == 0)
        {
          for (int j =120; j < 128; j++)
          {
            if (j % 2 == 0)
            {
              #if YANZHENG1
              fprintf(fp_luma, "%f ", fig1_AboveLeft_L.at(j, i)/255.0);
              #endif
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
#if YANZHENG1
              fprintf(fp_luma, "%f ", fig2_Left_L.at(j, i)/255.0);
#endif
              refenceLuma_Left_vec.push_back(fig2_Left_L.at(j, i) / maxvalue10);
            }
          }
        }
      }
      refrence_yLeft = torch::tensor(refenceLuma_Left_vec);
      refrence_yLeft = refrence_yLeft.view({ 1, 1, 64, 4 });   
      refrence_y     = torch::cat({ refrence_yAbove, refrence_yAboveLeft, refrence_yLeft }, 2); 
#if YANZHENG1
      fclose(fp_luma);
#endif
    }
   
    {   // U COMPONETNT
      PelBuf fig0_Above_L     = cs.getRecoBuf(cuAbove_C->block(COMPONENT_Cb));
      PelBuf fig1_AboveLeft_L = cs.getRecoBuf(cuAboveLeft_C->block(COMPONENT_Cb));
      PelBuf fig2_Left_L      = cs.getRecoBuf(cuLeft_C->block(COMPONENT_Cb));
      
      std::vector<float> refenceCb_Above_vec;
      std::vector<float> refenceCb_AboveLeft_vec;
      std::vector<float> refenceCb_Left_vec;
#if YANZHENG
     
      std::vector<float> refenceCb_Above_vec_YANZHENG;
      
      torch::Tensor      refrence_uAbove_YANZHENG;
      
#endif
      
     
      for (int i = 0; i < 64; i++)
      {
       
        for (int j = 60; j < 64; j++)
         {
           
              refenceCb_Above_vec.push_back(fig0_Above_L.at(j, i) / maxvalue10);
            
         }
        
      }
      refrence_uAbove = torch::tensor(refenceCb_Above_vec);
      
      refrence_uAbove = refrence_uAbove.view({ 4, 64 });
      refrence_uAbove = refrence_uAbove.t();
      #if YANZHENG1

      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 4; j++)
        {
          fprintf(fp_Cb, "%f ", refrence_uAbove[i][j].item().toFloat());
        }
      }

      #endif
      refrence_uAbove = refrence_uAbove.view({ 1, 1, 64, 4 });   //转置之后的正上方参考像素
#if YANZHENG
      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 64; j++)
        {
          refenceCb_Above_vec_YANZHENG.push_back(fig0_Above_L.at(j, i));
        }
      }
      refrence_uAbove_YANZHENG = torch::tensor(refenceCb_Above_vec_YANZHENG);
      refrence_uAbove_YANZHENG = refrence_uAbove_YANZHENG.view({ 64, 64 });
     
      //refrence_uAbove_YANZHENG = refrence_uAbove_YANZHENG.t();
      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 64; j++)
        {
          fprintf(fp_Cb, "%d ", refrence_uAbove_YANZHENG[i][j].item().toShort());
        }
      }
      fclose(fp_Cb);
#endif
      for (int i = 60; i < 64; i++)
      {
      
        
          for (int j = 60; j < 64; j++)
          {
            #if YANZHENG1
            fprintf(fp_Cb, "%f ", fig1_AboveLeft_L.at(j, i) / 255.0);
            #endif
            refenceCb_AboveLeft_vec.push_back(fig1_AboveLeft_L.at(j, i) / maxvalue10);
            
          }
      }
      
      refrence_uAboveLeft = torch::tensor(refenceCb_AboveLeft_vec);
      refrence_uAboveLeft = refrence_uAboveLeft.view({ 1, 1, 4, 4 });
      
        
        for (int i = 0; i < 64; i++)
        {
        
          for (int j = 60; j < 64; j++)
          {
           #if YANZHENG1
               fprintf(fp_Cb, "%f ", fig2_Left_L.at(j, i) / 255.0);
               #endif
               refenceCb_Left_vec.push_back(fig2_Left_L.at(j, i) / maxvalue10);
            
          }
        }
        
      refrence_uLeft = torch::tensor(refenceCb_Left_vec);
      refrence_uLeft = refrence_uLeft.view({ 1, 1, 64, 4 });
      refrence_u    = torch::cat({ refrence_uAbove, refrence_uAboveLeft, refrence_uLeft }, 2); 
     #if YANZHENG1
      fclose(fp_Cb);
      #endif
    }
   
    

     {   // V COMPONETNT
      PelBuf fig0_Above_L     = cs.getRecoBuf(cuAbove_C->block(COMPONENT_Cr));
      PelBuf fig1_AboveLeft_L = cs.getRecoBuf(cuAboveLeft_C->block(COMPONENT_Cr));
      PelBuf fig2_Left_L      = cs.getRecoBuf(cuLeft_C->block(COMPONENT_Cr));

      std::vector<float> refenceCr_Above_vec;
      std::vector<float> refenceCr_AboveLeft_vec;
      std::vector<float> refenceCr_Left_vec;
#if YANZHENG

      std::vector<float> refenceCr_Above_vec_YANZHENG;

      torch::Tensor refrence_vAbove_YANZHENG;

#endif

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
#if YANZHENG1

      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 4; j++)
        {
          fprintf(fp_Cr, "%f ", refrence_vAbove[i][j].item().toFloat());
        }
      }

#endif
      refrence_vAbove = refrence_vAbove.view({ 1, 1, 64, 4 });   //转置之后的正上方参考像素
#if YANZHENG
      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 64; j++)
        {
          refenceCr_Above_vec_YANZHENG.push_back(fig0_Above_L.at(j, i));
        }
      }
      refrence_vAbove_YANZHENG = torch::tensor(refenceCr_Above_vec_YANZHENG);
      refrence_vAbove_YANZHENG = refrence_vAbove_YANZHENG.view({ 64, 64 });
      //refrence_vAbove_YANZHENG = refrence_vAbove_YANZHENG.t();
      for (int i = 0; i < 64; i++)
      {
        for (int j = 0; j < 64; j++)
        {
          fprintf(fp_Cr, "%d ", refrence_vAbove_YANZHENG[i][j].item().toShort());
        }
      }
      fclose(fp_Cr);
#endif
      for (int i = 60; i < 64; i++)
      {
        for (int j = 60; j < 64; j++)
        {
#if YANZHENG1
          fprintf(fp_Cr, "%f ", fig1_AboveLeft_L.at(j, i) / 255.0);
#endif
          refenceCr_AboveLeft_vec.push_back(fig1_AboveLeft_L.at(j, i) / maxvalue10);
        }
      }

      refrence_vAboveLeft = torch::tensor(refenceCr_AboveLeft_vec);
      refrence_vAboveLeft = refrence_vAboveLeft.view({ 1, 1, 4, 4 });

      for (int i = 0; i < 64; i++)
      {
        for (int j = 60; j < 64; j++)
        {
#if YANZHENG1
          fprintf(fp_Cr, "%f ", fig2_Left_L.at(j, i) / 255.0);
#endif
          refenceCr_Left_vec.push_back(fig2_Left_L.at(j, i) / maxvalue10);
        }
      }

      refrence_vLeft = torch::tensor(refenceCr_Left_vec);
      refrence_vLeft = refrence_vLeft.view({ 1, 1, 64, 4 });
      refrence_v     = torch::cat({ refrence_vAbove, refrence_vAboveLeft, refrence_vLeft }, 2); 
#if YANZHENG1
      fclose(fp_Cr);
#endif
    }
    

    
  
    //Do the prediction
    refrence_yuv = torch::cat({ refrence_y, refrence_u, refrence_v }, 1);   //注意语法，要合并的tensor使用{}
    torch::jit::script::Module module;
    module = torch::jit::load("G:/可删神经网络参数/idea7gaijin1_128x128_59th.pt");

    
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
           #if YANZHENG1
          fprintf(fp_Cb1, "%d ", a);
          #endif
           
         }

         else
         {
           short b = True_output[0][k][uiY][uiX].item().toShort();
           #if YANZHENG1
           fprintf(fp_Cr1, "%d ",b );
           #endif
         }
         img_CTU_CNN[k][uiY * int(height / 2) + uiX] = True_output[0][k][uiY][uiX].item().toShort();
        
         
         
         
       }
     }
   }
   
   #if YANZHENG1 
  fclose(fp_Cb1);
  fclose(fp_Cr1);
  #endif
  }
  
  else
  {
     for (int k = 0; k < 2; k++)
    {
      for (int uiY = 0; uiY < int(width/2); uiY++)
      {
        for (int uiX = 0; uiX < int(height/2); uiX++)
        {
          img_CTU_CNN[k][uiY * int(height / 2) + uiX] = maxvalue10/2;
        }
      }
    }
  }
 
#endif
  if (CS::isDualITree (cs) && isChromaEnabled (cs.pcv->chrFormat))
  {
    m_CABACEstimator->getCtx() = m_CurrCtx->start;

    partitioner.initCtu(area, CH_C, *cs.slice);

    cs.initSubStructure(*tempCS, partitioner.chType, partitioner.currArea(), false);
    cs.initSubStructure(*bestCS, partitioner.chType, partitioner.currArea(), false);
    tempCS->currQP[CH_C] = bestCS->currQP[CH_C] =
    tempCS->baseQP       = bestCS->baseQP       = currQP[CH_C];
    tempCS->prevQP[CH_C] = bestCS->prevQP[CH_C] = prevQP[CH_C];

    xCompressCU(
#if ANNM
      flag1, img_CTU_CNN,//只要该块没有超过有边界和下边界,flag1=0
#endif
      tempCS, bestCS, partitioner);
#if testPSNR
    if (flag)
    {
      const CodingUnit *cuCurr_C   = bestCS->getCU(pos_C, CH_C);
      PelBuf fig_cur_Cb = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cb));
      PelBuf fig_cur_Cr = cs.getOrgBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            fig_cur_Cr_ped = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cr));
      PelBuf            fig_cur_Cb_ped = cs.getRecoBuf(cuCurr_C->block(COMPONENT_Cb));
      double U_MSEAVE  = 0;
      double U_PSNR = 0;
      double V_MSEAVE  = 0;
      double V_PSNR = 0;
      for (int uiY = 0; uiY < int(width / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(height / 2); uiX++)
        {
          double a = (fig_cur_Cr.at(uiX, uiY) - img_CTU_CNN[0][uiY * int(height / 2) + uiX])
                     * (fig_cur_Cr.at(uiX, uiY) - img_CTU_CNN[0][uiY * int(height / 2) + uiX]);
          //printf("a:%f \n", a);
          //printf("a/4096.0:%f \n", a/4096.0);
          U_MSEAVE = U_MSEAVE + a / 4096.0;
             
        }
      }
      U_PSNR = 10 * log10((maxvalue10 * maxvalue10 )/ U_MSEAVE);
      for (int uiY = 0; uiY < int(width / 2); uiY++)
      {
        for (int uiX = 0; uiX < int(height / 2); uiX++)
        {
          
          //double a = (fig_cur_Cr.at(uiX, uiY) - fig_cur_Cr_ped.at(uiX, uiY))
            //         * (fig_cur_Cr.at(uiX, uiY) - fig_cur_Cr_ped.at(uiX, uiY));
          double a = (fig_cur_Cr.at(uiX, uiY) - img_CTU_CNN[1][uiY * int(height / 2) + uiX])
                     * (fig_cur_Cr.at(uiX, uiY) - img_CTU_CNN[1][uiY * int(height / 2) + uiX]);
         //printf("b:%f \n", a);
         // printf("b/4096:%f \n", a/4096.0);
          V_MSEAVE = V_MSEAVE + a / 4096.0;
        }
      }
      V_PSNR = 10 * log10((maxvalue10 * maxvalue10) / V_MSEAVE);
      ////cout << U_PSNR << endl;
      ////cout << V_PSNR << endl;
      //printf("U_MSEAVE:%f\n", U_MSEAVE);
      //printf("V_MSEAVE:%f\n", V_MSEAVE);
       printf("%f , %f\n", U_PSNR,V_PSNR);
      //printf("x:%d  ,y:%d  ,V_PSNR:%f\n", bestCS->area.Y().lumaPos().x, bestCS->area.Y().lumaPos().y, V_PSNR);
    }
#endif
    const bool copyUnsplitCTUSignals = bestCS->cus.size() == 1;
    cs.useSubStructure(*bestCS, partitioner.chType, CS::getArea(*bestCS, area, partitioner.chType),
                       copyUnsplitCTUSignals, false, false, copyUnsplitCTUSignals);
  }

  if (m_pcEncCfg->getUseRateCtrl())
  {
    (m_pcRateCtrl->getRCPic()->getLCU(ctuRsAddr)).m_actualMSE = (double)bestCS->dist / (double)m_pcRateCtrl->getRCPic()->getLCU(ctuRsAddr).m_numberOfPixel;
  }
  // reset context states and uninit context pointer
  m_CABACEstimator->getCtx() = m_CurrCtx->start;
  m_CurrCtx                  = 0;


  // Ensure that a coding was found
  // Selected mode's RD-cost must be not MAX_DOUBLE.
  CHECK_VTM( bestCS->cus.empty()                                   , "No possible encoding found" );
  CHECK_VTM( bestCS->cus[0]->predMode == NUMBER_OF_PREDICTION_MODES, "No possible encoding found" );
  CHECK_VTM( bestCS->cost             == MAX_DOUBLE                , "No possible encoding found" );
#if PRINT
  // printf("(%d,%d) %d %d: sum bits is %lld\n", bestCS->area.blocks[0].x, bestCS->area.blocks[0].y,
  // bestCS->area.blocks[0].width, bestCS->area.blocks[0].height, bestCS->fracBits);
  static uint64_t sum = 0;
  sum += (bestCS->fracBits >> 15);
  printf("(%d,%d) %d %d: chroma bits is %lld\n", bestCS->area.blocks[1].x, bestCS->area.blocks[1].y,
         bestCS->area.blocks[1].width, bestCS->area.blocks[1].height, sum);
#endif
}

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

static int xCalcHADs8x8_ISlice(const Pel *piOrg, const int iStrideOrg)
{
  int k, i, j, jj;
  int diff[64], m1[8][8], m2[8][8], m3[8][8], iSumHad = 0;

  for (k = 0; k < 64; k += 8)
  {
    diff[k + 0] = piOrg[0];
    diff[k + 1] = piOrg[1];
    diff[k + 2] = piOrg[2];
    diff[k + 3] = piOrg[3];
    diff[k + 4] = piOrg[4];
    diff[k + 5] = piOrg[5];
    diff[k + 6] = piOrg[6];
    diff[k + 7] = piOrg[7];

    piOrg += iStrideOrg;
  }

  //horizontal
  for (j = 0; j < 8; j++)
  {
    jj = j << 3;
    m2[j][0] = diff[jj    ] + diff[jj + 4];
    m2[j][1] = diff[jj + 1] + diff[jj + 5];
    m2[j][2] = diff[jj + 2] + diff[jj + 6];
    m2[j][3] = diff[jj + 3] + diff[jj + 7];
    m2[j][4] = diff[jj    ] - diff[jj + 4];
    m2[j][5] = diff[jj + 1] - diff[jj + 5];
    m2[j][6] = diff[jj + 2] - diff[jj + 6];
    m2[j][7] = diff[jj + 3] - diff[jj + 7];

    m1[j][0] = m2[j][0] + m2[j][2];
    m1[j][1] = m2[j][1] + m2[j][3];
    m1[j][2] = m2[j][0] - m2[j][2];
    m1[j][3] = m2[j][1] - m2[j][3];
    m1[j][4] = m2[j][4] + m2[j][6];
    m1[j][5] = m2[j][5] + m2[j][7];
    m1[j][6] = m2[j][4] - m2[j][6];
    m1[j][7] = m2[j][5] - m2[j][7];

    m2[j][0] = m1[j][0] + m1[j][1];
    m2[j][1] = m1[j][0] - m1[j][1];
    m2[j][2] = m1[j][2] + m1[j][3];
    m2[j][3] = m1[j][2] - m1[j][3];
    m2[j][4] = m1[j][4] + m1[j][5];
    m2[j][5] = m1[j][4] - m1[j][5];
    m2[j][6] = m1[j][6] + m1[j][7];
    m2[j][7] = m1[j][6] - m1[j][7];
  }

  //vertical
  for (i = 0; i < 8; i++)
  {
    m3[0][i] = m2[0][i] + m2[4][i];
    m3[1][i] = m2[1][i] + m2[5][i];
    m3[2][i] = m2[2][i] + m2[6][i];
    m3[3][i] = m2[3][i] + m2[7][i];
    m3[4][i] = m2[0][i] - m2[4][i];
    m3[5][i] = m2[1][i] - m2[5][i];
    m3[6][i] = m2[2][i] - m2[6][i];
    m3[7][i] = m2[3][i] - m2[7][i];

    m1[0][i] = m3[0][i] + m3[2][i];
    m1[1][i] = m3[1][i] + m3[3][i];
    m1[2][i] = m3[0][i] - m3[2][i];
    m1[3][i] = m3[1][i] - m3[3][i];
    m1[4][i] = m3[4][i] + m3[6][i];
    m1[5][i] = m3[5][i] + m3[7][i];
    m1[6][i] = m3[4][i] - m3[6][i];
    m1[7][i] = m3[5][i] - m3[7][i];

    m2[0][i] = m1[0][i] + m1[1][i];
    m2[1][i] = m1[0][i] - m1[1][i];
    m2[2][i] = m1[2][i] + m1[3][i];
    m2[3][i] = m1[2][i] - m1[3][i];
    m2[4][i] = m1[4][i] + m1[5][i];
    m2[5][i] = m1[4][i] - m1[5][i];
    m2[6][i] = m1[6][i] + m1[7][i];
    m2[7][i] = m1[6][i] - m1[7][i];
  }

  for (i = 0; i < 8; i++)
  {
    for (j = 0; j < 8; j++)
    {
      iSumHad += abs(m2[i][j]);
    }
  }
  iSumHad -= abs(m2[0][0]);
  iSumHad = (iSumHad + 2) >> 2;
  return(iSumHad);
}

int  EncCu::updateCtuDataISlice(const CPelBuf buf)
{
  int  xBl, yBl;
  const int iBlkSize = 8;
  const Pel* pOrgInit = buf.buf;
  int  iStrideOrig = buf.stride;

  int iSumHad = 0;
  for( yBl = 0; ( yBl + iBlkSize ) <= buf.height; yBl += iBlkSize )
  {
    for( xBl = 0; ( xBl + iBlkSize ) <= buf.width; xBl += iBlkSize )
    {
      const Pel* pOrg = pOrgInit + iStrideOrig*yBl + xBl;
      iSumHad += xCalcHADs8x8_ISlice( pOrg, iStrideOrig );
    }
  }
  return( iSumHad );
}

bool EncCu::xCheckBestMode( CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode& encTestMode )
{
  bool bestCSUpdated = false;

  if( !tempCS->cus.empty() )
  {
    if( tempCS->cus.size() == 1 )
    {
      const CodingUnit& cu = *tempCS->cus.front();
      CHECK_VTM( cu.skip && !cu.firstPU->mergeFlag, "Skip flag without a merge flag is not allowed!" );
    }

#if WCG_EXT
    DTRACE_BEST_MODE( tempCS, bestCS, m_pcRdCost->getLambda( true ) );
#else
    DTRACE_BEST_MODE( tempCS, bestCS, m_pcRdCost->getLambda() );
#endif

    if( m_modeCtrl->useModeResult( encTestMode, tempCS, partitioner ) )
    {

      std::swap( tempCS, bestCS );
      // store temp best CI for next CU coding
      m_CurrCtx->best = m_CABACEstimator->getCtx();
      m_bestModeUpdated = true;
      bestCSUpdated = true;
    }
  }

  // reset context states
  m_CABACEstimator->getCtx() = m_CurrCtx->start;
  return bestCSUpdated;

}

void EncCu::xCompressCU(
#if ANNM
  bool &flag, Pel *img[],
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, double maxCostAllowed)
{
  CHECK_VTM(maxCostAllowed < 0, "Wrong value of maxCostAllowed!");
#if ENABLE_SPLIT_PARALLELISM
  CHECK_VTM( m_dataId != tempCS->picture->scheduler.getDataId(), "Working in the wrong dataId!" );

  if( m_pcEncCfg->getNumSplitThreads() != 1 && tempCS->picture->scheduler.getSplitJobId() == 0 )
  {
    if( m_modeCtrl->isParallelSplit( *tempCS, partitioner ) )
    {
      m_modeCtrl->setParallelSplit( true );
      xCompressCUParallel( tempCS, bestCS, partitioner );
      return;
    }
  }

#endif
  uint32_t compBegin;
  uint32_t numComp;
  bool jointPLT = false;
  if (partitioner.isSepTree( *tempCS ))
  {
    if( !CS::isDualITree(*tempCS) && partitioner.treeType != TREE_D )
    {
      compBegin = COMPONENT_Y;
      numComp = (tempCS->area.chromaFormat != CHROMA_400)?3: 1;
      jointPLT = true;
    }
    else
    {
    if (isLuma(partitioner.chType))
    {
      compBegin = COMPONENT_Y;
      numComp = 1;
    }
    else
    {
      compBegin = COMPONENT_Cb;
      numComp = 2;
    }
    }
  }
  else
  {
    compBegin = COMPONENT_Y;
    numComp = (tempCS->area.chromaFormat != CHROMA_400) ? 3 : 1;
    jointPLT = true;
  }
  SplitSeries splitmode = -1;
  uint8_t   bestLastPLTSize[MAX_NUM_CHANNEL_TYPE];
  Pel       bestLastPLT[MAX_NUM_COMPONENT][MAXPLTPREDSIZE]; // store LastPLT for
  uint8_t   curLastPLTSize[MAX_NUM_CHANNEL_TYPE];
  Pel       curLastPLT[MAX_NUM_COMPONENT][MAXPLTPREDSIZE]; // store LastPLT if no partition
  for (int i = compBegin; i < (compBegin + numComp); i++)
  {
    ComponentID comID = jointPLT ? (ComponentID)compBegin : ((i > 0) ? COMPONENT_Cb : COMPONENT_Y);
    bestLastPLTSize[comID] = 0;
    curLastPLTSize[comID] = tempCS->prevPLT.curPLTSize[comID];
    memcpy(curLastPLT[i], tempCS->prevPLT.curPLT[i], tempCS->prevPLT.curPLTSize[comID] * sizeof(Pel));
  }

  Slice&   slice      = *tempCS->slice;
  const PPS &pps      = *tempCS->pps;
  const SPS &sps      = *tempCS->sps;
  const uint32_t uiLPelX  = tempCS->area.Y().lumaPos().x;
  const uint32_t uiTPelY  = tempCS->area.Y().lumaPos().y;
#if ANNM


  const uint32_t uiWidth  = tempCS->area.Cb().chromaSize().width;
  const uint32_t uiHeight = tempCS->area.Cb().chromaSize().height;

  const uint32_t uiLPelX0 = tempCS->area.Cb().chromaPos().x;
  const uint32_t uiTPelY0 = tempCS->area.Cb().chromaPos().y;   
#endif
  const ModeType modeTypeParent  = partitioner.modeType;
  const TreeType treeTypeParent  = partitioner.treeType;
  const ChannelType chTypeParent = partitioner.chType;
  const UnitArea currCsArea = clipArea( CS::getArea( *bestCS, bestCS->area, partitioner.chType ), *tempCS->picture );

  m_modeCtrl->initCULevel( partitioner, *tempCS );
  if( partitioner.currQtDepth == 0 && partitioner.currMtDepth == 0 && !tempCS->slice->isIntra() && ( sps.getUseSBT() || sps.getUseInterMTS() ) )
  {
    auto slsSbt = dynamic_cast<SaveLoadEncInfoSbt*>( m_modeCtrl );
    int maxSLSize = sps.getUseSBT() ? tempCS->slice->getSPS()->getMaxTbSize() : MTS_INTER_MAX_CU_SIZE;
    slsSbt->resetSaveloadSbt( maxSLSize );
#if ENABLE_SPLIT_PARALLELISM
    CHECK_VTM( tempCS->picture->scheduler.getSplitJobId() != 0, "The SBT search reset need to happen in sequential region." );
    if (m_pcEncCfg->getNumSplitThreads() > 1)
    {
      for (int jId = 1; jId < NUM_RESERVERD_SPLIT_JOBS; jId++)
      {
        auto slsSbt = dynamic_cast<SaveLoadEncInfoSbt *>(m_pcEncLib->getCuEncoder(jId)->m_modeCtrl);
        slsSbt->resetSaveloadSbt(maxSLSize);
      }
    }
#endif
  }
  m_sbtCostSave[0] = m_sbtCostSave[1] = MAX_DOUBLE;

  m_CurrCtx->start = m_CABACEstimator->getCtx();

  m_cuChromaQpOffsetIdxPlus1 = 0;

  if( slice.getUseChromaQpAdj() )
  {
    // TODO M0133 : double check encoder decisions with respect to chroma QG detection and actual encode
    int lgMinCuSize = sps.getLog2MinCodingBlockSize() +
      std::max<int>(0, floorLog2(sps.getCTUSize()) - sps.getLog2MinCodingBlockSize() - int(slice.getCuChromaQpOffsetSubdiv() / 2));
    if( partitioner.currQgChromaEnable() )
    {
      m_cuChromaQpOffsetIdxPlus1 = ( ( uiLPelX >> lgMinCuSize ) + ( uiTPelY >> lgMinCuSize ) ) % ( pps.getChromaQpOffsetListLen() + 1 );
    }
  }

  if( !m_modeCtrl->anyMode() )
  {
    m_modeCtrl->finishCULevel( partitioner );
    return;
  }

  DTRACE_UPDATE( g_trace_ctx, std::make_pair( "cux", uiLPelX ) );
  DTRACE_UPDATE( g_trace_ctx, std::make_pair( "cuy", uiTPelY ) );
  DTRACE_UPDATE( g_trace_ctx, std::make_pair( "cuw", tempCS->area.lwidth() ) );
  DTRACE_UPDATE( g_trace_ctx, std::make_pair( "cuh", tempCS->area.lheight() ) );
  DTRACE( g_trace_ctx, D_COMMON, "@(%4d,%4d) [%2dx%2d]\n", tempCS->area.lx(), tempCS->area.ly(), tempCS->area.lwidth(), tempCS->area.lheight() );


  m_pcInterSearch->resetSavedAffineMotion();

  double bestIntPelCost = MAX_DOUBLE;

  if (tempCS->slice->getSPS()->getUseColorTrans())
  {
    tempCS->tmpColorSpaceCost = MAX_DOUBLE;
    bestCS->tmpColorSpaceCost = MAX_DOUBLE;
    tempCS->firstColorSpaceSelected = true;
    bestCS->firstColorSpaceSelected = true;
  }

  if (tempCS->slice->getSPS()->getUseColorTrans() && !CS::isDualITree(*tempCS))
  {
    tempCS->firstColorSpaceTestOnly = false;
    bestCS->firstColorSpaceTestOnly = false;
    tempCS->tmpColorSpaceIntraCost[0] = MAX_DOUBLE;
    tempCS->tmpColorSpaceIntraCost[1] = MAX_DOUBLE;
    bestCS->tmpColorSpaceIntraCost[0] = MAX_DOUBLE;
    bestCS->tmpColorSpaceIntraCost[1] = MAX_DOUBLE;

    if (tempCS->bestParent && tempCS->bestParent->firstColorSpaceTestOnly)
    {
      tempCS->firstColorSpaceTestOnly = bestCS->firstColorSpaceTestOnly = true;
    }
  }

  do
  {
    for (int i = compBegin; i < (compBegin + numComp); i++)
    {
      ComponentID comID = jointPLT ? (ComponentID)compBegin : ((i > 0) ? COMPONENT_Cb : COMPONENT_Y);
      tempCS->prevPLT.curPLTSize[comID] = curLastPLTSize[comID];
      memcpy(tempCS->prevPLT.curPLT[i], curLastPLT[i], curLastPLTSize[comID] * sizeof(Pel));
    }
    EncTestMode currTestMode = m_modeCtrl->currTestMode();
    currTestMode.maxCostAllowed = maxCostAllowed;

    if (pps.getUseDQP() && partitioner.isSepTree(*tempCS) && isChroma( partitioner.chType ))
    {
      const Position chromaCentral(tempCS->area.Cb().chromaPos().offset(tempCS->area.Cb().chromaSize().width >> 1, tempCS->area.Cb().chromaSize().height >> 1));
      const Position lumaRefPos(chromaCentral.x << getComponentScaleX(COMPONENT_Cb, tempCS->area.chromaFormat), chromaCentral.y << getComponentScaleY(COMPONENT_Cb, tempCS->area.chromaFormat));
      const CodingStructure* baseCS = bestCS->picture->cs;
      const CodingUnit* colLumaCu = baseCS->getCU(lumaRefPos, CHANNEL_TYPE_LUMA);

      if (colLumaCu)
      {
        currTestMode.qp = colLumaCu->qp;
      }
    }

#if SHARP_LUMA_DELTA_QP || ENABLE_QPA_SUB_CTU
    if (partitioner.currQgEnable() && (
#if SHARP_LUMA_DELTA_QP
        (m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled()) ||
#endif
#if ENABLE_QPA_SUB_CTU
        (m_pcEncCfg->getUsePerceptQPA() && !m_pcEncCfg->getUseRateCtrl() && pps.getUseDQP())
#else
        false
#endif
      ))
    {
#if ENABLE_SPLIT_PARALLELISM
      CHECK_VTM( tempCS->picture->scheduler.getSplitJobId() > 0, "Changing lambda is only allowed in the master thread!" );
#endif
      if (currTestMode.qp >= 0)
      {
        updateLambda (&slice, currTestMode.qp,
 #if WCG_EXT && ER_CHROMA_QP_WCG_PPS
                      m_pcEncCfg->getWCGChromaQPControl().isEnabled(),
 #endif
                      CS::isDualITree (*tempCS) || (partitioner.currDepth == 0));
      }
    }
#endif

    if( currTestMode.type == ETM_INTER_ME )
    {
      if( ( currTestMode.opts & ETO_IMV ) != 0 )
      {
        const bool skipAltHpelIF = ( int( ( currTestMode.opts & ETO_IMV ) >> ETO_IMV_SHIFT ) == 4 ) && ( bestIntPelCost > 1.25 * bestCS->cost );
        if (!skipAltHpelIF)
        {
          tempCS->bestCS = bestCS;
          xCheckRDCostInterIMV(
#if ANNM
            flag,
#endif
            tempCS, bestCS, partitioner, currTestMode, bestIntPelCost);
          tempCS->bestCS = nullptr;
        }
      }
      else
      {
        tempCS->bestCS = bestCS;
        xCheckRDCostInter(
#if ANNM
            flag,
#endif
          tempCS, bestCS, partitioner, currTestMode);
        tempCS->bestCS = nullptr;
      }

    }
    else if (currTestMode.type == ETM_HASH_INTER)
    {
      xCheckRDCostHashInter(
#if ANNM
            flag,
#endif
        tempCS, bestCS, partitioner, currTestMode);
    }
    else if( currTestMode.type == ETM_AFFINE )
    {
      xCheckRDCostAffineMerge2Nx2N( 
#if ANNM
            flag,
#endif
        tempCS, bestCS, partitioner, currTestMode);
    }
#if REUSE_CU_RESULTS
    else if( currTestMode.type == ETM_RECO_CACHED )
    {
      xReuseCachedResult( 
#if ANNM
        img, uiLPelX0, uiTPelY0,
#endif
        tempCS, bestCS, partitioner);
    }
#endif
    else if( currTestMode.type == ETM_MERGE_SKIP )
    {
      xCheckRDCostMerge2Nx2N( 
#if ANNM
        flag, img, uiLPelX0, uiTPelY0,
#endif
        tempCS, bestCS, partitioner, currTestMode);
      CodingUnit* cu = bestCS->getCU(partitioner.chType);
      if (cu)
      cu->mmvdSkip = cu->skip == false ? false : cu->mmvdSkip;
    }
    else if( currTestMode.type == ETM_MERGE_GEO )
    {
      xCheckRDCostMergeGeo2Nx2N( 
#if ANNM
            flag,
#endif
        tempCS, bestCS, partitioner, currTestMode);
    }
    else if( currTestMode.type == ETM_INTRA )
    {
      if (slice.getSPS()->getUseColorTrans() && !CS::isDualITree(*tempCS))
      {
        bool skipSecColorSpace = false;
        skipSecColorSpace = xCheckRDCostIntra(
#if ANNM
        flag, img, uiLPelX0, uiTPelY0,
#endif
          tempCS, bestCS, partitioner, currTestMode, (m_pcEncCfg->getRGBFormatFlag() ? true : false));
        if ((m_pcEncCfg->getCostMode() == COST_LOSSLESS_CODING) && !m_pcEncCfg->getRGBFormatFlag())
        {
          skipSecColorSpace = true;
        }
        if (!skipSecColorSpace && !tempCS->firstColorSpaceTestOnly)
        {
          xCheckRDCostIntra(
#if ANNM
        flag, img, uiLPelX0, uiTPelY0,
#endif
            tempCS, bestCS, partitioner, currTestMode, (m_pcEncCfg->getRGBFormatFlag() ? false : true));
        }

        if (!tempCS->firstColorSpaceTestOnly)
        {
          if (tempCS->tmpColorSpaceIntraCost[0] != MAX_DOUBLE && tempCS->tmpColorSpaceIntraCost[1] != MAX_DOUBLE)
          {
            double skipCostRatio = m_pcEncCfg->getRGBFormatFlag() ? 1.1 : 1.0;
            if (tempCS->tmpColorSpaceIntraCost[1] > (skipCostRatio*tempCS->tmpColorSpaceIntraCost[0]))
            {
              tempCS->firstColorSpaceTestOnly = bestCS->firstColorSpaceTestOnly = true;
            }
          }
        }
        else
        {
          CHECK_VTM(tempCS->tmpColorSpaceIntraCost[1] != MAX_DOUBLE, "the RD test of the second color space should be skipped");
        }
      }
      else
      {
        xCheckRDCostIntra(
#if ANNM
        flag, img, uiLPelX0, uiTPelY0,
#endif
          tempCS, bestCS, partitioner, currTestMode, false);
      }
    }
    else if (currTestMode.type == ETM_PALETTE)
    {
      xCheckPLT( tempCS, bestCS, partitioner, currTestMode );
    }
    else if (currTestMode.type == ETM_IBC)
    {
      xCheckRDCostIBCMode(
#if ANNM
        flag,
#endif
        tempCS, bestCS, partitioner, currTestMode);
    }
    else if (currTestMode.type == ETM_IBC_MERGE)
    {
      xCheckRDCostIBCModeMerge2Nx2N(
#if ANNM
        flag,
#endif
        tempCS, bestCS, partitioner, currTestMode);
    }
    else if( isModeSplit( currTestMode ) )
    {
      if (bestCS->cus.size() != 0)
      {
        splitmode = bestCS->cus[0]->splitSeries;
      }
      assert( partitioner.modeType == tempCS->modeType );
      int signalModeConsVal = tempCS->signalModeCons( getPartSplit( currTestMode ), partitioner, modeTypeParent );
      int numRoundRdo = signalModeConsVal == LDT_MODE_TYPE_SIGNAL ? 2 : 1;
      bool skipInterPass = false;
      for( int i = 0; i < numRoundRdo; i++ )
      {
        //change cons modes
        if( signalModeConsVal == LDT_MODE_TYPE_SIGNAL )
        {
          CHECK_VTM( numRoundRdo != 2, "numRoundRdo shall be 2 - [LDT_MODE_TYPE_SIGNAL]" );
          tempCS->modeType = partitioner.modeType = (i == 0) ? MODE_TYPE_INTER : MODE_TYPE_INTRA;
        }
        else if( signalModeConsVal == LDT_MODE_TYPE_INFER )
        {
          CHECK_VTM( numRoundRdo != 1, "numRoundRdo shall be 1 - [LDT_MODE_TYPE_INFER]" );
          tempCS->modeType = partitioner.modeType = MODE_TYPE_INTRA;
        }
        else if( signalModeConsVal == LDT_MODE_TYPE_INHERIT )
        {
          CHECK_VTM( numRoundRdo != 1, "numRoundRdo shall be 1 - [LDT_MODE_TYPE_INHERIT]" );
          tempCS->modeType = partitioner.modeType = modeTypeParent;
        }

        //for lite intra encoding fast algorithm, set the status to save inter coding info
        if( modeTypeParent == MODE_TYPE_ALL && tempCS->modeType == MODE_TYPE_INTER )
        {
          m_pcIntraSearch->setSaveCuCostInSCIPU( true );
          m_pcIntraSearch->setNumCuInSCIPU( 0 );
        }
        else if( modeTypeParent == MODE_TYPE_ALL && tempCS->modeType != MODE_TYPE_INTER )
        {
          m_pcIntraSearch->setSaveCuCostInSCIPU( false );
          if( tempCS->modeType == MODE_TYPE_ALL )
          {
            m_pcIntraSearch->setNumCuInSCIPU( 0 );
          }
        }

        xCheckModeSplit( 
#if ANNM
          flag, img,
#endif
          tempCS, bestCS, partitioner, currTestMode, modeTypeParent, skipInterPass );
        //recover cons modes
        tempCS->modeType = partitioner.modeType = modeTypeParent;
        tempCS->treeType = partitioner.treeType = treeTypeParent;
        partitioner.chType = chTypeParent;
        if( modeTypeParent == MODE_TYPE_ALL )
        {
          m_pcIntraSearch->setSaveCuCostInSCIPU( false );
          if( numRoundRdo == 2 && tempCS->modeType == MODE_TYPE_INTRA )
          {
            m_pcIntraSearch->initCuAreaCostInSCIPU();
          }
        }
        if( skipInterPass )
        {
          break;
        }
      }
      if (splitmode != bestCS->cus[0]->splitSeries)
      {
        splitmode = bestCS->cus[0]->splitSeries;
        const CodingUnit&     cu = *bestCS->cus.front();
        cu.cs->prevPLT = bestCS->prevPLT;
        for (int i = compBegin; i < (compBegin + numComp); i++)
        {
          ComponentID comID = jointPLT ? (ComponentID)compBegin : ((i > 0) ? COMPONENT_Cb : COMPONENT_Y);
          bestLastPLTSize[comID] = bestCS->cus[0]->cs->prevPLT.curPLTSize[comID];
          memcpy(bestLastPLT[i], bestCS->cus[0]->cs->prevPLT.curPLT[i], bestCS->cus[0]->cs->prevPLT.curPLTSize[comID] * sizeof(Pel));
        }
      }
    }
    else
    {
      THROW( "Don't know how to handle mode: type = " << currTestMode.type << ", options = " << currTestMode.opts );
    }
  } while( m_modeCtrl->nextMode( *tempCS, partitioner ) );


  //////////////////////////////////////////////////////////////////////////
  // Finishing CU
#if ENABLE_SPLIT_PARALLELISM
  if( bestCS->cus.empty() )
  {
    CHECK_VTM( bestCS->cost != MAX_DOUBLE, "Cost should be maximal if no encoding found" );
    CHECK_VTM( bestCS->picture->scheduler.getSplitJobId() == 0, "Should always get a result in serial case" );

    m_modeCtrl->finishCULevel( partitioner );
    return;
  }

#endif
  if( tempCS->cost == MAX_DOUBLE && bestCS->cost == MAX_DOUBLE )
  {
    //although some coding modes were planned to be tried in RDO, no coding mode actually finished encoding due to early termination
    //thus tempCS->cost and bestCS->cost are both MAX_DOUBLE; in this case, skip the following process for normal case
    m_modeCtrl->finishCULevel( partitioner );
    return;
  }

  // set context states
  m_CABACEstimator->getCtx() = m_CurrCtx->best;

  // QP from last processed CU for further processing
  //copy the qp of the last non-chroma CU
  int numCUInThisNode = (int)bestCS->cus.size();
  if( numCUInThisNode > 1 && bestCS->cus.back()->chType == CHANNEL_TYPE_CHROMA && !CS::isDualITree( *bestCS ) )
  {
    CHECK_VTM( bestCS->cus[numCUInThisNode-2]->chType != CHANNEL_TYPE_LUMA, "wrong chType" );
    bestCS->prevQP[partitioner.chType] = bestCS->cus[numCUInThisNode-2]->qp;
  }
  else
  {
  bestCS->prevQP[partitioner.chType] = bestCS->cus.back()->qp;
  }
  if ((!slice.isIntra() || slice.getSPS()->getIBCFlag())
    && partitioner.chType == CHANNEL_TYPE_LUMA
    && bestCS->cus.size() == 1 && (bestCS->cus.back()->predMode == MODE_INTER || bestCS->cus.back()->predMode == MODE_IBC)
    && bestCS->area.Y() == (*bestCS->cus.back()).Y()
    )
  {
    const CodingUnit&     cu = *bestCS->cus.front();

    bool isIbcSmallBlk = CU::isIBC(cu) && (cu.lwidth() * cu.lheight() <= 16);
    CU::saveMotionInHMVP( cu, isIbcSmallBlk );
  }
  bestCS->picture->getPredBuf(currCsArea).copyFrom(bestCS->getPredBuf(currCsArea));
  bestCS->picture->getRecoBuf( currCsArea ).copyFrom( bestCS->getRecoBuf( currCsArea ) );
  m_modeCtrl->finishCULevel( partitioner );
  if( m_pcIntraSearch->getSaveCuCostInSCIPU() && bestCS->cus.size() == 1 )
  {
    m_pcIntraSearch->saveCuAreaCostInSCIPU( Area( partitioner.currArea().lumaPos(), partitioner.currArea().lumaSize() ), bestCS->cost );
  }

#if ENABLE_SPLIT_PARALLELISM
  if( tempCS->picture->scheduler.getSplitJobId() == 0 && m_pcEncCfg->getNumSplitThreads() != 1 )
  {
    tempCS->picture->finishParallelPart( currCsArea );
  }

#endif
  if (bestCS->cus.size() == 1) // no partition
  {
    CHECK_VTM(bestCS->cus[0]->tileIdx != bestCS->pps->getTileIdx(bestCS->area.lumaPos()), "Wrong tile index!");
    if (bestCS->cus[0]->predMode == MODE_PLT)
    {
      for (int i = compBegin; i < (compBegin + numComp); i++)
      {
        ComponentID comID = jointPLT ? (ComponentID)compBegin : ((i > 0) ? COMPONENT_Cb : COMPONENT_Y);
        bestCS->prevPLT.curPLTSize[comID] = curLastPLTSize[comID];
        memcpy(bestCS->prevPLT.curPLT[i], curLastPLT[i], curLastPLTSize[comID] * sizeof(Pel));
      }
      bestCS->reorderPrevPLT(bestCS->prevPLT, bestCS->cus[0]->curPLTSize, bestCS->cus[0]->curPLT, bestCS->cus[0]->reuseflag, compBegin, numComp, jointPLT);
    }
    else
    {
      for (int i = compBegin; i<(compBegin + numComp); i++)
      {
        ComponentID comID = jointPLT ? (ComponentID)compBegin : ((i > 0) ? COMPONENT_Cb : COMPONENT_Y);
        bestCS->prevPLT.curPLTSize[comID] = curLastPLTSize[comID];
        memcpy(bestCS->prevPLT.curPLT[i], curLastPLT[i], bestCS->prevPLT.curPLTSize[comID] * sizeof(Pel));
      }
    }
  }
  else
  {
    for (int i = compBegin; i<(compBegin + numComp); i++)
    {
      ComponentID comID = jointPLT ? (ComponentID)compBegin : ((i > 0) ? COMPONENT_Cb : COMPONENT_Y);
      bestCS->prevPLT.curPLTSize[comID] = bestLastPLTSize[comID];
      memcpy(bestCS->prevPLT.curPLT[i], bestLastPLT[i], bestCS->prevPLT.curPLTSize[comID] * sizeof(Pel));
    }
  }
  const CodingUnit&     cu = *bestCS->cus.front();
  cu.cs->prevPLT = bestCS->prevPLT;
  // Assert if Best prediction mode is NONE
  // Selected mode's RD-cost must be not MAX_DOUBLE.
  CHECK_VTM( bestCS->cus.empty()                                   , "No possible encoding found" );
  CHECK_VTM( bestCS->cus[0]->predMode == NUMBER_OF_PREDICTION_MODES, "No possible encoding found" );
  CHECK_VTM( bestCS->cost             == MAX_DOUBLE                , "No possible encoding found" );
}

#if SHARP_LUMA_DELTA_QP || ENABLE_QPA_SUB_CTU
void EncCu::updateLambda (Slice* slice, const int dQP,
 #if WCG_EXT && ER_CHROMA_QP_WCG_PPS
                          const bool useWCGChromaControl,
 #endif
                          const bool updateRdCostLambda)
{
#if WCG_EXT && ER_CHROMA_QP_WCG_PPS
  if (useWCGChromaControl)
  {
    const double lambda = m_pcSliceEncoder->initializeLambda (slice, m_pcSliceEncoder->getGopId(), slice->getSliceQp(), (double)dQP);
    const int clippedQP = Clip3 (-slice->getSPS()->getQpBDOffset (CHANNEL_TYPE_LUMA), MAX_QP, dQP);

    m_pcSliceEncoder->setUpLambda (slice, lambda, clippedQP);
    return;
  }
#endif
  int iQP = dQP;
  const double oldQP     = (double)slice->getSliceQpBase();
#if ENABLE_QPA_SUB_CTU
  const double oldLambda = (m_pcEncCfg->getUsePerceptQPA() && !m_pcEncCfg->getUseRateCtrl() && slice->getPPS()->getUseDQP()) ? slice->getLambdas()[0] :
                           m_pcSliceEncoder->calculateLambda (slice, m_pcSliceEncoder->getGopId(), oldQP, oldQP, iQP);
#else
  const double oldLambda = m_pcSliceEncoder->calculateLambda (slice, m_pcSliceEncoder->getGopId(), oldQP, oldQP, iQP);
#endif
  const double newLambda = oldLambda * pow (2.0, ((double)dQP - oldQP) / 3.0);
#if RDOQ_CHROMA_LAMBDA
  const double lambdaArray[MAX_NUM_COMPONENT] = {newLambda / m_pcRdCost->getDistortionWeight (COMPONENT_Y),
                                                 newLambda / m_pcRdCost->getDistortionWeight (COMPONENT_Cb),
                                                 newLambda / m_pcRdCost->getDistortionWeight (COMPONENT_Cr)};
  m_pcTrQuant->setLambdas (lambdaArray);
#else
  m_pcTrQuant->setLambda (newLambda);
#endif
  if (updateRdCostLambda)
  {
    m_pcRdCost->setLambda (newLambda, slice->getSPS()->getBitDepths());
#if WCG_EXT
    m_pcRdCost->saveUnadjustedLambda();
#endif
  }
}
#endif // SHARP_LUMA_DELTA_QP || ENABLE_QPA_SUB_CTU

#if ENABLE_SPLIT_PARALLELISM
//#undef DEBUG_PARALLEL_TIMINGS
//#define DEBUG_PARALLEL_TIMINGS 1
void EncCu::xCompressCUParallel( CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner )
{
  const unsigned wIdx = gp_sizeIdxInfo->idxFrom( partitioner.currArea().lwidth() );
  const unsigned hIdx = gp_sizeIdxInfo->idxFrom( partitioner.currArea().lheight() );

  Picture* picture = tempCS->picture;

  int numJobs = m_modeCtrl->getNumParallelJobs( *bestCS, partitioner );

  bool    jobUsed                            [NUM_RESERVERD_SPLIT_JOBS];
  std::fill( jobUsed, jobUsed + NUM_RESERVERD_SPLIT_JOBS, false );

  const UnitArea currArea = CS::getArea( *tempCS, partitioner.currArea(), partitioner.chType );
  const bool doParallel   = !m_pcEncCfg->getForceSingleSplitThread();
  omp_set_num_threads( m_pcEncCfg->getNumSplitThreads() );

#pragma omp parallel for schedule(dynamic,1) if(doParallel)
  for( int jId = 1; jId <= numJobs; jId++ )
  {
    // thread start
    picture->scheduler.setSplitThreadId();
    picture->scheduler.setSplitJobId( jId );

    QTBTPartitioner jobPartitioner;
    EncCu*       jobCuEnc       = m_pcEncLib->getCuEncoder( picture->scheduler.getSplitDataId( jId ) );
    auto*        jobBlkCache    = dynamic_cast<CacheBlkInfoCtrl*>( jobCuEnc->m_modeCtrl );
#if REUSE_CU_RESULTS
    auto*        jobBestCache   = dynamic_cast<BestEncInfoCache*>( jobCuEnc->m_modeCtrl );
#endif

    jobPartitioner.copyState( partitioner );
    jobCuEnc      ->copyState( this, jobPartitioner, currArea, true );

    if( jobBlkCache  ) { jobBlkCache ->tick(); }
#if REUSE_CU_RESULTS
    if( jobBestCache ) { jobBestCache->tick(); }

#endif
    CodingStructure *&jobBest = jobCuEnc->m_pBestCS[wIdx][hIdx];
    CodingStructure *&jobTemp = jobCuEnc->m_pTempCS[wIdx][hIdx];

    jobUsed[jId] = true;

    jobCuEnc->xCompressCU( jobTemp, jobBest, jobPartitioner );

    picture->scheduler.setSplitJobId( 0 );
    // thread stop
  }
  picture->scheduler.setSplitThreadId( 0 );

  int    bestJId  = 0;
  double bestCost = bestCS->cost;
  for( int jId = 1; jId <= numJobs; jId++ )
  {
    EncCu* jobCuEnc = m_pcEncLib->getCuEncoder( picture->scheduler.getSplitDataId( jId ) );

    if( jobUsed[jId] && jobCuEnc->m_pBestCS[wIdx][hIdx]->cost < bestCost )
    {
      bestCost = jobCuEnc->m_pBestCS[wIdx][hIdx]->cost;
      bestJId  = jId;
    }
  }

  if( bestJId > 0 )
  {
    copyState( m_pcEncLib->getCuEncoder( picture->scheduler.getSplitDataId( bestJId ) ), partitioner, currArea, false );
    m_CurrCtx->best = m_CABACEstimator->getCtx();

    tempCS = m_pTempCS[wIdx][hIdx];
    bestCS = m_pBestCS[wIdx][hIdx];
  }

  const int      bitDepthY = tempCS->sps->getBitDepth( CH_L );
  const UnitArea clipdArea = clipArea( currArea, *picture );

  CHECK_VTM( calcCheckSum( picture->getRecoBuf( clipdArea.Y() ), bitDepthY ) != calcCheckSum( bestCS->getRecoBuf( clipdArea.Y() ), bitDepthY ), "Data copied incorrectly!" );

  picture->finishParallelPart( currArea );

  if( auto *blkCache = dynamic_cast<CacheBlkInfoCtrl*>( m_modeCtrl ) )
  {
    for( int jId = 1; jId <= numJobs; jId++ )
    {
      if( !jobUsed[jId] || jId == bestJId ) continue;

      auto *jobBlkCache = dynamic_cast<CacheBlkInfoCtrl*>( m_pcEncLib->getCuEncoder( picture->scheduler.getSplitDataId( jId ) )->m_modeCtrl );
      CHECK_VTM( !jobBlkCache, "If own mode controller has blk info cache capability so should all other mode controllers!" );
      blkCache->CacheBlkInfoCtrl::copyState( *jobBlkCache, partitioner.currArea() );
    }

    blkCache->tick();
  }
#if REUSE_CU_RESULTS

  if( auto *blkCache = dynamic_cast<BestEncInfoCache*>( m_modeCtrl ) )
  {
    for( int jId = 1; jId <= numJobs; jId++ )
    {
      if( !jobUsed[jId] || jId == bestJId ) continue;

      auto *jobBlkCache = dynamic_cast<BestEncInfoCache*>( m_pcEncLib->getCuEncoder( picture->scheduler.getSplitDataId( jId ) )->m_modeCtrl );
      CHECK_VTM( !jobBlkCache, "If own mode controller has blk info cache capability so should all other mode controllers!" );
      blkCache->BestEncInfoCache::copyState( *jobBlkCache, partitioner.currArea() );
    }

    blkCache->tick();
  }
#endif
}

void EncCu::copyState( EncCu* other, Partitioner& partitioner, const UnitArea& currArea, const bool isDist )
{
  const unsigned wIdx = gp_sizeIdxInfo->idxFrom( partitioner.currArea().lwidth () );
  const unsigned hIdx = gp_sizeIdxInfo->idxFrom( partitioner.currArea().lheight() );

  if( isDist )
  {
    other->m_pBestCS[wIdx][hIdx]->initSubStructure( *m_pBestCS[wIdx][hIdx], partitioner.chType, partitioner.currArea(), false );
    other->m_pTempCS[wIdx][hIdx]->initSubStructure( *m_pTempCS[wIdx][hIdx], partitioner.chType, partitioner.currArea(), false );
  }
  else
  {
          CodingStructure* dst =        m_pBestCS[wIdx][hIdx];
    const CodingStructure* src = other->m_pBestCS[wIdx][hIdx];
    bool keepResi = KEEP_PRED_AND_RESI_SIGNALS;
    bool keepPred = true;

    dst->useSubStructure( *src, partitioner.chType, currArea, keepPred, true, keepResi, keepResi );

    dst->cost           =  src->cost;
    dst->dist           =  src->dist;
    dst->fracBits       =  src->fracBits;
    dst->features       =  src->features;
  }

  if( isDist )
  {
    m_CurrCtx = m_CtxBuffer.data();
  }

  m_pcInterSearch->copyState( *other->m_pcInterSearch );
  m_modeCtrl     ->copyState( *other->m_modeCtrl, partitioner.currArea() );
  m_pcRdCost     ->copyState( *other->m_pcRdCost );
  m_pcTrQuant    ->copyState( *other->m_pcTrQuant );
  if( m_pcEncCfg->getLmcs() )
  {
    EncReshape *encReshapeThis  = dynamic_cast<EncReshape*>(       m_pcReshape);
    EncReshape *encReshapeOther = dynamic_cast<EncReshape*>(other->m_pcReshape);
    encReshapeThis->copyState( *encReshapeOther );
  }

  m_CABACEstimator->getCtx() = other->m_CABACEstimator->getCtx();
}
#endif

void EncCu::xCheckModeSplit(
#if ANNM
  bool& flag, Pel* img[],
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode,    \
  const ModeType modeTypeParent, bool &skipInterPass)
{
  const int qp                = encTestMode.qp;
  const Slice &slice          = *tempCS->slice;
  const int oldPrevQp         = tempCS->prevQP[partitioner.chType];
  const auto oldMotionLut     = tempCS->motionLut;
#if ENABLE_QPA_SUB_CTU
  const PPS &pps              = *tempCS->pps;
  const uint32_t currDepth    = partitioner.currDepth;
#endif
  const auto oldPLT           = tempCS->prevPLT;

  const PartSplit split = getPartSplit( encTestMode );
  const ModeType modeTypeChild = partitioner.modeType;

  CHECK_VTM( split == CU_DONT_SPLIT, "No proper split provided!" );

  tempCS->initStructData( qp );

  m_CABACEstimator->getCtx() = m_CurrCtx->start;

  const TempCtx ctxStartSP( m_CtxCache, SubCtx( Ctx::SplitFlag,   m_CABACEstimator->getCtx() ) );
  const TempCtx ctxStartQt( m_CtxCache, SubCtx( Ctx::SplitQtFlag, m_CABACEstimator->getCtx() ) );
  const TempCtx ctxStartHv( m_CtxCache, SubCtx( Ctx::SplitHvFlag, m_CABACEstimator->getCtx() ) );
  const TempCtx ctxStart12( m_CtxCache, SubCtx( Ctx::Split12Flag, m_CABACEstimator->getCtx() ) );
  const TempCtx ctxStartMC( m_CtxCache, SubCtx( Ctx::ModeConsFlag, m_CABACEstimator->getCtx() ) );
  m_CABACEstimator->resetBits();

  m_CABACEstimator->split_cu_mode( split, *tempCS, partitioner );
  m_CABACEstimator->mode_constraint( split, *tempCS, partitioner, modeTypeChild );

  const double factor = ( tempCS->currQP[partitioner.chType] > 30 ? 1.1 : 1.075 );
  tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();
  if (!tempCS->useDbCost)
    CHECK_VTM(bestCS->costDbOffset != 0, "error");
  const double cost   = m_pcRdCost->calcRdCost( uint64_t( m_CABACEstimator->getEstFracBits() + ( ( bestCS->fracBits ) / factor ) ), Distortion( bestCS->dist / factor ) ) + bestCS->costDbOffset / factor;

  m_CABACEstimator->getCtx() = SubCtx( Ctx::SplitFlag,   ctxStartSP );
  m_CABACEstimator->getCtx() = SubCtx( Ctx::SplitQtFlag, ctxStartQt );
  m_CABACEstimator->getCtx() = SubCtx( Ctx::SplitHvFlag, ctxStartHv );
  m_CABACEstimator->getCtx() = SubCtx( Ctx::Split12Flag, ctxStart12 );
  m_CABACEstimator->getCtx() = SubCtx( Ctx::ModeConsFlag, ctxStartMC );
  if (cost > bestCS->cost + bestCS->costDbOffset
#if ENABLE_QPA_SUB_CTU
    || (m_pcEncCfg->getUsePerceptQPA() && !m_pcEncCfg->getUseRateCtrl() && pps.getUseDQP() && (slice.getCuQpDeltaSubdiv() > 0) && (split == CU_HORZ_SPLIT || split == CU_VERT_SPLIT) &&
        (currDepth == 0)) // force quad-split or no split at CTU level
#endif
    )
  {
    xCheckBestMode( tempCS, bestCS, partitioner, encTestMode );
    return;
  }

  const bool chromaNotSplit = modeTypeParent == MODE_TYPE_ALL && modeTypeChild == MODE_TYPE_INTRA ? true : false;
  if( partitioner.treeType != TREE_D )
  {
    tempCS->treeType = TREE_L;
  }
  else
  {
    if( chromaNotSplit )
    {
      CHECK_VTM( partitioner.chType != CHANNEL_TYPE_LUMA, "chType must be luma" );
      tempCS->treeType = partitioner.treeType = TREE_L;
    }
    else
    {
      tempCS->treeType = partitioner.treeType = TREE_D;
    }
  }


  partitioner.splitCurrArea( split, *tempCS );
  bool qgEnableChildren = partitioner.currQgEnable(); // QG possible at children level

  m_CurrCtx++;

  tempCS->getRecoBuf().fill( 0 );

  tempCS->getPredBuf().fill(0);
  AffineMVInfo tmpMVInfo;
  bool isAffMVInfoSaved;
  m_pcInterSearch->savePrevAffMVInfo(0, tmpMVInfo, isAffMVInfoSaved);
  BlkUniMvInfo tmpUniMvInfo;
  bool         isUniMvInfoSaved = false;
  if (!tempCS->slice->isIntra())
  {
    m_pcInterSearch->savePrevUniMvInfo(tempCS->area.Y(), tmpUniMvInfo, isUniMvInfoSaved);
  }

  do
  {
    const auto &subCUArea  = partitioner.currArea();

    if( tempCS->picture->Y().contains( subCUArea.lumaPos() ) )
    {
      const unsigned wIdx    = gp_sizeIdxInfo->idxFrom( subCUArea.lwidth () );
      const unsigned hIdx    = gp_sizeIdxInfo->idxFrom( subCUArea.lheight() );

      CodingStructure *tempSubCS = m_pTempCS[wIdx][hIdx];
      CodingStructure *bestSubCS = m_pBestCS[wIdx][hIdx];

      tempCS->initSubStructure( *tempSubCS, partitioner.chType, subCUArea, false );
      tempCS->initSubStructure( *bestSubCS, partitioner.chType, subCUArea, false );
      tempSubCS->bestParent = bestSubCS->bestParent = bestCS;
      double newMaxCostAllowed = isLuma(partitioner.chType) ? std::min(encTestMode.maxCostAllowed, bestCS->cost - m_pcRdCost->calcRdCost(tempCS->fracBits, tempCS->dist)) : MAX_DOUBLE;
      newMaxCostAllowed = std::max(0.0, newMaxCostAllowed);
      xCompressCU(
#if ANNM
        flag, img,
#endif
        tempSubCS, bestSubCS, partitioner, newMaxCostAllowed);
      tempSubCS->bestParent = bestSubCS->bestParent = nullptr;

      if( bestSubCS->cost == MAX_DOUBLE )
      {
        CHECK_VTM( split == CU_QUAD_SPLIT, "Split decision reusing cannot skip quad split" );
        tempCS->cost = MAX_DOUBLE;
        tempCS->costDbOffset = 0;
        tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();
        m_CurrCtx--;
        partitioner.exitCurrSplit();
        xCheckBestMode( tempCS, bestCS, partitioner, encTestMode );
        if( partitioner.chType == CHANNEL_TYPE_LUMA )
        {
          tempCS->motionLut = oldMotionLut;
        }
        return;
      }

      bool keepResi = KEEP_PRED_AND_RESI_SIGNALS;
      tempCS->useSubStructure( *bestSubCS, partitioner.chType, CS::getArea( *tempCS, subCUArea, partitioner.chType ), KEEP_PRED_AND_RESI_SIGNALS, true, keepResi, keepResi );

      if( partitioner.currQgEnable() )
      {
        tempCS->prevQP[partitioner.chType] = bestSubCS->prevQP[partitioner.chType];
      }
      if( partitioner.isConsInter() )
      {
        for( int i = 0; i < bestSubCS->cus.size(); i++ )
        {
          CHECK_VTM( bestSubCS->cus[i]->predMode != MODE_INTER, "all CUs must be inter mode in an Inter coding region (SCIPU)" );
        }
      }
      else if( partitioner.isConsIntra() )
      {
        for( int i = 0; i < bestSubCS->cus.size(); i++ )
        {
          CHECK_VTM( bestSubCS->cus[i]->predMode == MODE_INTER, "all CUs must not be inter mode in an Intra coding region (SCIPU)" );
        }
      }

      tempSubCS->releaseIntermediateData();
      bestSubCS->releaseIntermediateData();
#if SPLIT_SHOW
      tempSubCS->partype = encTestMode.type;
      bestSubCS->partype = encTestMode.type;
#endif
      if( !tempCS->slice->isIntra() && partitioner.isConsIntra() )
      {
        tempCS->cost = m_pcRdCost->calcRdCost( tempCS->fracBits, tempCS->dist );
        if( tempCS->cost > bestCS->cost )
        {
          tempCS->cost = MAX_DOUBLE;
          tempCS->costDbOffset = 0;
          tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();
          m_CurrCtx--;
          partitioner.exitCurrSplit();
          if( partitioner.chType == CHANNEL_TYPE_LUMA )
          {
            tempCS->motionLut = oldMotionLut;
          }
          return;
        }
      }
    }
  } while( partitioner.nextPart( *tempCS ) );

  partitioner.exitCurrSplit();


  m_CurrCtx--;

  if( chromaNotSplit )
  {
    //Note: In local dual tree region, the chroma CU refers to the central luma CU's QP.
    //If the luma CU QP shall be predQP (no residual in it and before it in the QG), it must be revised to predQP before encoding the chroma CU
    //Otherwise, the chroma CU uses predQP+deltaQP in encoding but is decoded as using predQP, thus causing encoder-decoded mismatch on chroma qp.
    if( tempCS->pps->getUseDQP() )
    {
      //find parent CS that including all coded CUs in the QG before this node
      CodingStructure* qgCS = tempCS;
      bool deltaQpCodedBeforeThisNode = false;
      if( partitioner.currArea().lumaPos() != partitioner.currQgPos )
      {
        int numParentNodeToQgCS = 0;
        while( qgCS->area.lumaPos() != partitioner.currQgPos )
        {
          CHECK_VTM( qgCS->parent == nullptr, "parent of qgCS shall exsit" );
          qgCS = qgCS->parent;
          numParentNodeToQgCS++;
        }

        //check whether deltaQP has been coded (in luma CU or luma&chroma CU) before this node
        CodingStructure* parentCS = tempCS->parent;
        for( int i = 0; i < numParentNodeToQgCS; i++ )
        {
          //checking each parent
          CHECK_VTM( parentCS == nullptr, "parentCS shall exsit" );
          for( const auto &cu : parentCS->cus )
          {
            if( cu->rootCbf && !isChroma( cu->chType ) )
            {
              deltaQpCodedBeforeThisNode = true;
              break;
            }
          }
          parentCS = parentCS->parent;
        }
      }

      //revise luma CU qp before the first luma CU with residual in the SCIPU to predQP
      if( !deltaQpCodedBeforeThisNode )
      {
        //get pred QP of the QG
        const CodingUnit* cuFirst = qgCS->getCU( CHANNEL_TYPE_LUMA );
        CHECK_VTM( cuFirst->lumaPos() != partitioner.currQgPos, "First cu of the Qg is wrong" );
        int predQp = CU::predictQP( *cuFirst, qgCS->prevQP[CHANNEL_TYPE_LUMA] );

        //revise to predQP
        int firstCuHasResidual = (int)tempCS->cus.size();
        for( int i = 0; i < tempCS->cus.size(); i++ )
        {
          if( tempCS->cus[i]->rootCbf )
          {
            firstCuHasResidual = i;
            break;
          }
        }

        for( int i = 0; i < firstCuHasResidual; i++ )
        {
          tempCS->cus[i]->qp = predQp;
        }
      }
    }
    assert( tempCS->treeType == TREE_L );
    uint32_t numCuPuTu[6];
    tempCS->picture->cs->getNumCuPuTuOffset( numCuPuTu );
    tempCS->picture->cs->useSubStructure( *tempCS, partitioner.chType, CS::getArea( *tempCS, partitioner.currArea(), partitioner.chType ), false, true, false, false );

    if (isChromaEnabled(tempCS->pcv->chrFormat))
    {
    partitioner.chType = CHANNEL_TYPE_CHROMA;
    tempCS->treeType = partitioner.treeType = TREE_C;

    m_CurrCtx++;

    const unsigned wIdx = gp_sizeIdxInfo->idxFrom( partitioner.currArea().lwidth() );
    const unsigned hIdx = gp_sizeIdxInfo->idxFrom( partitioner.currArea().lheight() );
    CodingStructure *tempCSChroma = m_pTempCS2[wIdx][hIdx];
    CodingStructure *bestCSChroma = m_pBestCS2[wIdx][hIdx];
    tempCS->initSubStructure( *tempCSChroma, partitioner.chType, partitioner.currArea(), false );
    tempCS->initSubStructure( *bestCSChroma, partitioner.chType, partitioner.currArea(), false );
    tempCS->treeType = TREE_D;
    xCompressCU( 
#if ANNM
        flag, img,
#endif
      tempCSChroma, bestCSChroma, partitioner);

    //attach chromaCS to luma CS and update cost
    bool keepResi = KEEP_PRED_AND_RESI_SIGNALS;
    //bestCSChroma->treeType = tempCSChroma->treeType = TREE_C;
    CHECK_VTM( bestCSChroma->treeType != TREE_C || tempCSChroma->treeType != TREE_C, "wrong treeType for chroma CS" );
    tempCS->useSubStructure( *bestCSChroma, partitioner.chType, CS::getArea( *bestCSChroma, partitioner.currArea(), partitioner.chType ), KEEP_PRED_AND_RESI_SIGNALS, true, keepResi, true );

    //release tmp resource
    tempCSChroma->releaseIntermediateData();
    bestCSChroma->releaseIntermediateData();
    //tempCS->picture->cs->releaseIntermediateData();
      m_CurrCtx--;
    }
    tempCS->picture->cs->clearCuPuTuIdxMap( partitioner.currArea(), numCuPuTu[0], numCuPuTu[1], numCuPuTu[2], numCuPuTu + 3 );


    //recover luma tree status
    partitioner.chType = CHANNEL_TYPE_LUMA;
    partitioner.treeType = TREE_D;
    partitioner.modeType = MODE_TYPE_ALL;
  }

  // Finally, generate split-signaling bits for RD-cost check
  const PartSplit implicitSplit = partitioner.getImplicitSplit( *tempCS );

  {
    bool enforceQT = implicitSplit == CU_QUAD_SPLIT;

    // LARGE CTU bug
    if( m_pcEncCfg->getUseFastLCTU() )
    {
      unsigned minDepth = 0;
      unsigned maxDepth = floorLog2(tempCS->sps->getCTUSize()) - floorLog2(tempCS->sps->getMinQTSize(slice.getSliceType(), partitioner.chType));

      if( auto ad = dynamic_cast<AdaptiveDepthPartitioner*>( &partitioner ) )
      {
        ad->setMaxMinDepth( minDepth, maxDepth, *tempCS );
      }

      if( minDepth > partitioner.currQtDepth )
      {
        // enforce QT
        enforceQT = true;
      }
    }

    if( !enforceQT )
    {
      m_CABACEstimator->resetBits();

      m_CABACEstimator->split_cu_mode( split, *tempCS, partitioner );
      partitioner.modeType = modeTypeParent;
      m_CABACEstimator->mode_constraint( split, *tempCS, partitioner, modeTypeChild );
      tempCS->fracBits += m_CABACEstimator->getEstFracBits(); // split bits
    }
  }

  tempCS->cost = m_pcRdCost->calcRdCost( tempCS->fracBits, tempCS->dist );

  // Check Delta QP bits for splitted structure
  if( !qgEnableChildren ) // check at deepest QG level only
  xCheckDQP( *tempCS, partitioner, true );

  // If the configuration being tested exceeds the maximum number of bytes for a slice / slice-segment, then
  // a proper RD evaluation cannot be performed. Therefore, termination of the
  // slice/slice-segment must be made prior to this CTU.
  // This can be achieved by forcing the decision to be that of the rpcTempCU.
  // The exception is each slice / slice-segment must have at least one CTU.
  if (bestCS->cost != MAX_DOUBLE)
  {
  }
  else
  {
    bestCS->costDbOffset = 0;
  }
  tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();
  if( tempCS->cus.size() > 0 && modeTypeParent == MODE_TYPE_ALL && modeTypeChild == MODE_TYPE_INTER )
  {
    int areaSizeNoResiCu = 0;
    for( int k = 0; k < tempCS->cus.size(); k++ )
    {
      areaSizeNoResiCu += (tempCS->cus[k]->rootCbf == false) ? tempCS->cus[k]->lumaSize().area() : 0;
    }
    if( areaSizeNoResiCu >= (tempCS->area.lumaSize().area() >> 1) )
    {
      skipInterPass = true;
    }
  }

  // RD check for sub partitioned coding structure.
  xCheckBestMode( tempCS, bestCS, partitioner, encTestMode );

  if (isAffMVInfoSaved)
    m_pcInterSearch->addAffMVInfo(tmpMVInfo);
  if (!tempCS->slice->isIntra() && isUniMvInfoSaved)
  {
    m_pcInterSearch->addUniMvInfo(tmpUniMvInfo);
  }

  tempCS->motionLut = oldMotionLut;

  tempCS->prevPLT   = oldPLT;

  tempCS->releaseIntermediateData();

  tempCS->prevQP[partitioner.chType] = oldPrevQp;
}

bool EncCu::xCheckRDCostIntra(
#if ANNM
  bool& flag, Pel* img[], int pieX, int pieY,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode,    \
  bool adaptiveColorTrans)
{
  double          bestInterCost             = m_modeCtrl->getBestInterCost();
  double          costSize2Nx2NmtsFirstPass = m_modeCtrl->getMtsSize2Nx2NFirstPassCost();
  bool            skipSecondMtsPass         = m_modeCtrl->getSkipSecondMTSPass();
  const SPS&      sps                       = *tempCS->sps;
  const int       maxSizeMTS                = MTS_INTRA_MAX_CU_SIZE;
  uint8_t         considerMtsSecondPass     = ( sps.getUseIntraMTS() && isLuma( partitioner.chType ) && partitioner.currArea().lwidth() <= maxSizeMTS && partitioner.currArea().lheight() <= maxSizeMTS ) ? 1 : 0;

  bool   useIntraSubPartitions   = false;
  double maxCostAllowedForChroma = MAX_DOUBLE;
  const  CodingUnit *bestCU      = bestCS->getCU( partitioner.chType );
  Distortion interHad = m_modeCtrl->getInterHad();


  double dct2Cost                =   MAX_DOUBLE;
  double bestNonDCT2Cost         = MAX_DOUBLE;
  double trGrpBestCost     [ 4 ] = { MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE };
  double globalBestCost          =   MAX_DOUBLE;
  bool   bestSelFlag       [ 4 ] = { false, false, false, false };
  bool   trGrpCheck        [ 4 ] = { true, true, true, true };
  int    startMTSIdx       [ 4 ] = { 0, 1, 2, 3 };
  int    endMTSIdx         [ 4 ] = { 0, 1, 2, 3 };
  double trGrpStopThreshold[ 3 ] = { 1.001, 1.001, 1.001 };
  int    bestMtsFlag             =   0;
  int    bestLfnstIdx            =   0;

  const int  maxLfnstIdx         = ( partitioner.isSepTree( *tempCS ) && partitioner.chType == CHANNEL_TYPE_CHROMA && ( partitioner.currArea().lwidth() < 8 || partitioner.currArea().lheight() < 8 ) )
                                   || ( partitioner.currArea().lwidth() > sps.getMaxTbSize() || partitioner.currArea().lheight() > sps.getMaxTbSize() ) ? 0 : 2;
  bool       skipOtherLfnst      = false;
  int        startLfnstIdx       = 0;
  int        endLfnstIdx         = sps.getUseLFNST() ? maxLfnstIdx : 0;

  int grpNumMax = sps.getUseLFNST() ? m_pcEncCfg->getMTSIntraMaxCand() : 1;
  m_modeCtrl->setISPWasTested(false);
  m_pcIntraSearch->invalidateBestModeCost();
  if (sps.getUseColorTrans() && !CS::isDualITree(*tempCS))
  {
    if ((m_pcEncCfg->getRGBFormatFlag() && adaptiveColorTrans) || (!m_pcEncCfg->getRGBFormatFlag() && !adaptiveColorTrans))
    {
      m_pcIntraSearch->invalidateBestRdModeFirstColorSpace();
    }
  }

  bool foundZeroRootCbf = false;
  if (sps.getUseColorTrans())
  {
    CHECK_VTM(tempCS->treeType != TREE_D || partitioner.treeType != TREE_D, "localtree should not be applied when adaptive color transform is enabled");
    CHECK_VTM(tempCS->modeType != MODE_TYPE_ALL || partitioner.modeType != MODE_TYPE_ALL, "localtree should not be applied when adaptive color transform is enabled");
    CHECK_VTM(adaptiveColorTrans && (CS::isDualITree(*tempCS) || partitioner.chType != CHANNEL_TYPE_LUMA), "adaptive color transform cannot be applied to dual-tree");
  }

  for( int trGrpIdx = 0; trGrpIdx < grpNumMax; trGrpIdx++ )
  {
    const uint8_t startMtsFlag = trGrpIdx > 0;
    const uint8_t endMtsFlag   = sps.getUseLFNST() ? considerMtsSecondPass : 0;

    if( ( trGrpIdx == 0 || ( !skipSecondMtsPass && considerMtsSecondPass ) ) && trGrpCheck[ trGrpIdx ] )
    {
      for( int lfnstIdx = startLfnstIdx; lfnstIdx <= endLfnstIdx; lfnstIdx++ )
      {
        for( uint8_t mtsFlag = startMtsFlag; mtsFlag <= endMtsFlag; mtsFlag++ )
        {
          if (sps.getUseColorTrans() && !CS::isDualITree(*tempCS))
          {
            m_pcIntraSearch->setSavedRdModeIdx(trGrpIdx*(NUM_LFNST_NUM_PER_SET * 2) + lfnstIdx * 2 + mtsFlag);
          }
          if (mtsFlag > 0 && lfnstIdx > 0)
          {
            continue;
          }
          //3) if interHad is 0, only try further modes if some intra mode was already better than inter
          if( sps.getUseLFNST() && m_pcEncCfg->getUsePbIntraFast() && !tempCS->slice->isIntra() && bestCU && CU::isInter( *bestCS->getCU( partitioner.chType ) ) && interHad == 0 )
          {
            continue;
          }

          tempCS->initStructData( encTestMode.qp );

          CodingUnit &cu      = tempCS->addCU( CS::getArea( *tempCS, tempCS->area, partitioner.chType ), partitioner.chType );

          partitioner.setCUData( cu );
          cu.slice            = tempCS->slice;
          cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
          cu.skip             = false;
          cu.mmvdSkip = false;
          cu.predMode         = MODE_INTRA;
          cu.chromaQpAdj      = m_cuChromaQpOffsetIdxPlus1;
          cu.qp               = encTestMode.qp;
          cu.lfnstIdx         = lfnstIdx;
          cu.mtsFlag          = mtsFlag;
          cu.ispMode          = NOT_INTRA_SUBPARTITIONS;
          cu.colorTransform = adaptiveColorTrans;

          CU::addPUs( cu );

          tempCS->interHad    = interHad;

          m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;

          bool validCandRet = false;
          if( isLuma( partitioner.chType ) )
          {
            //ISP uses the value of the best cost so far (luma if it is the fast version) to avoid test non-necessary subpartitions
            double bestCostSoFar = partitioner.isSepTree(*tempCS) ? m_modeCtrl->getBestCostWithoutSplitFlags() : bestCU && bestCU->predMode == MODE_INTRA ? bestCS->lumaCost : bestCS->cost;
            if (partitioner.isSepTree(*tempCS) && encTestMode.maxCostAllowed < bestCostSoFar)
            {
              bestCostSoFar = encTestMode.maxCostAllowed;
            }
            validCandRet = m_pcIntraSearch->estIntraPredLumaQT(
#if ANNM
        flag, img, pieX, pieY,
#endif
              cu, partitioner, bestCostSoFar, mtsFlag,                                                                         \
  startMTSIdx[trGrpIdx], endMTSIdx[trGrpIdx], (trGrpIdx > 0), !cu.colorTransform ? bestCS : nullptr);
            if ((!validCandRet || (cu.ispMode && cu.firstTU->cbf[COMPONENT_Y] == 0)))
            {
              continue;
            }
            if (m_pcEncCfg->getUseFastISP() && validCandRet && !mtsFlag && !lfnstIdx && !cu.colorTransform)
            {
              m_modeCtrl->setISPMode(cu.ispMode);
              m_modeCtrl->setISPLfnstIdx(cu.lfnstIdx);
              m_modeCtrl->setMIPFlagISPPass(cu.mipFlag);
              m_modeCtrl->setBestISPIntraModeRelCU(cu.ispMode ? PU::getFinalIntraMode(*cu.firstPU, CHANNEL_TYPE_LUMA) : UINT8_MAX);
              m_modeCtrl->setBestDCT2NonISPCostRelCU(m_modeCtrl->getMtsFirstPassNoIspCost());
            }

            if (sps.getUseColorTrans() && m_pcEncCfg->getRGBFormatFlag() && !CS::isDualITree(*tempCS) && !cu.colorTransform)
            {
              double curLumaCost = m_pcRdCost->calcRdCost(tempCS->fracBits, tempCS->dist);
              if (curLumaCost > bestCS->cost)
              {
                continue;
              }
            }

            useIntraSubPartitions = cu.ispMode != NOT_INTRA_SUBPARTITIONS;
            if( !partitioner.isSepTree( *tempCS ) )
            {
              tempCS->lumaCost = m_pcRdCost->calcRdCost( tempCS->fracBits, tempCS->dist );
              if( useIntraSubPartitions )
              {
                //the difference between the best cost so far and the current luma cost is stored to avoid testing the Cr component if the cost of luma + Cb is larger than the best cost
                maxCostAllowedForChroma = bestCS->cost < MAX_DOUBLE ? bestCS->cost - tempCS->lumaCost : MAX_DOUBLE;
              }
            }

            if (m_pcEncCfg->getUsePbIntraFast() && tempCS->dist == std::numeric_limits<Distortion>::max()
                && tempCS->interHad == 0)
            {
              interHad = 0;
              // JEM assumes only perfect reconstructions can from now on beat the inter mode
              m_modeCtrl->enforceInterHad( 0 );
              continue;
            }

            if( !partitioner.isSepTree( *tempCS ) )
            {
              if (!cu.colorTransform)
              {
                cu.cs->picture->getRecoBuf(cu.Y()).copyFrom(cu.cs->getRecoBuf(COMPONENT_Y));
                cu.cs->picture->getPredBuf(cu.Y()).copyFrom(cu.cs->getPredBuf(COMPONENT_Y));
              }
              else
              {
                cu.cs->picture->getRecoBuf(cu).copyFrom(cu.cs->getRecoBuf(cu));
                cu.cs->picture->getPredBuf(cu).copyFrom(cu.cs->getPredBuf(cu));
              }
            }
          }

          if( tempCS->area.chromaFormat != CHROMA_400 && ( partitioner.chType == CHANNEL_TYPE_CHROMA || !cu.isSepTree() ) && !cu.colorTransform )
          {
            TUIntraSubPartitioner subTuPartitioner( partitioner );
            m_pcIntraSearch->estIntraPredChromaQT(
#if ANNM
        flag, img, pieX, pieY,
#endif
              cu,                                                                                                              \
  (!useIntraSubPartitions || (cu.isSepTree() && !isLuma(CHANNEL_TYPE_CHROMA))) ? partitioner : subTuPartitioner,       \
  maxCostAllowedForChroma);
            if( useIntraSubPartitions && !cu.ispMode )
            {
              //At this point the temp cost is larger than the best cost. Therefore, we can already skip the remaining calculations
              continue;
            }
          }

          cu.rootCbf = false;

          for( uint32_t t = 0; t < getNumberValidTBlocks( *cu.cs->pcv ); t++ )
          {
            cu.rootCbf |= cu.firstTU->cbf[t] != 0;
          }

          if (!cu.rootCbf)
          {
            cu.colorTransform = false;
            foundZeroRootCbf = true;
          }

          // Get total bits for current mode: encode CU
          m_CABACEstimator->resetBits();

          if ((!cu.cs->slice->isIntra() || cu.cs->slice->getSPS()->getIBCFlag())
            && cu.Y().valid()
            )
          {
            m_CABACEstimator->cu_skip_flag ( cu );
          }
          m_CABACEstimator->pred_mode      ( cu );
          m_CABACEstimator->adaptive_color_transform(cu);
          m_CABACEstimator->cu_pred_data   (
#if ANNM
      flag,
#endif
            cu);

          // Encode Coefficients
          CUCtx cuCtx;
          cuCtx.isDQPCoded = true;
          cuCtx.isChromaQpAdjCoded = true;
          m_CABACEstimator->cu_residual( cu, partitioner, cuCtx );

          tempCS->fracBits = m_CABACEstimator->getEstFracBits();
          tempCS->cost     = m_pcRdCost->calcRdCost(tempCS->fracBits, tempCS->dist);


          const double tmpCostWithoutSplitFlags = tempCS->cost;
          xEncodeDontSplit( *tempCS, partitioner );

          xCheckDQP( *tempCS, partitioner );
          xCheckChromaQPOffset( *tempCS, partitioner );

          // Check if low frequency non-separable transform (LFNST) is too expensive
          if( lfnstIdx && !cuCtx.lfnstLastScanPos && !cu.ispMode )
          {
            bool cbfAtZeroDepth = cu.isSepTree() ?
                                       cu.rootCbf
                                     : (tempCS->area.chromaFormat != CHROMA_400 && std::min( cu.firstTU->blocks[ 1 ].width, cu.firstTU->blocks[ 1 ].height ) < 4) ?
                                            TU::getCbfAtDepth( *cu.firstTU, COMPONENT_Y, 0 )
                                          : cu.rootCbf;
            if( cbfAtZeroDepth )
            {
              tempCS->cost = MAX_DOUBLE;
            }
          }

          if (isLuma(partitioner.chType) && cu.firstTU->mtsIdx[COMPONENT_Y] > MTS_SKIP)
          {
            CHECK_VTM(!cuCtx.mtsLastScanPos, "MTS is disallowed to only contain DC coefficient");
          }

          if( mtsFlag == 0 && lfnstIdx == 0 )
          {
            dct2Cost = tempCS->cost;
          }
          else if (tmpCostWithoutSplitFlags < bestNonDCT2Cost)
          {
            bestNonDCT2Cost = tmpCostWithoutSplitFlags;
          }

          if( tempCS->cost < bestCS->cost )
          {
            m_modeCtrl->setBestCostWithoutSplitFlags( tmpCostWithoutSplitFlags );
          }

          if( !mtsFlag ) static_cast< double& >( costSize2Nx2NmtsFirstPass ) = tempCS->cost;

          if( sps.getUseLFNST() && !tempCS->cus.empty() )
          {
            skipOtherLfnst = m_modeCtrl->checkSkipOtherLfnst( encTestMode, tempCS, partitioner );
          }

          xCalDebCost( *tempCS, partitioner );
          tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();


#if WCG_EXT
          DTRACE_MODE_COST( *tempCS, m_pcRdCost->getLambda( true ) );
#else
          DTRACE_MODE_COST( *tempCS, m_pcRdCost->getLambda() );
#endif
          if (sps.getUseColorTrans() && !CS::isDualITree(*tempCS))
          {
            int colorSpaceIdx = ((m_pcEncCfg->getRGBFormatFlag() && adaptiveColorTrans) || (!m_pcEncCfg->getRGBFormatFlag() && !adaptiveColorTrans)) ? 0 : 1;
            if (tempCS->cost < tempCS->tmpColorSpaceIntraCost[colorSpaceIdx])
            {
              tempCS->tmpColorSpaceIntraCost[colorSpaceIdx] = tempCS->cost;
              bestCS->tmpColorSpaceIntraCost[colorSpaceIdx] = tempCS->cost;
            }
          }
          if( !sps.getUseLFNST() )
          {
            xCheckBestMode( tempCS, bestCS, partitioner, encTestMode );
          }
          else
          {
            if( xCheckBestMode( tempCS, bestCS, partitioner, encTestMode ) )
            {
              trGrpBestCost[ trGrpIdx ] = globalBestCost = bestCS->cost;
              bestSelFlag  [ trGrpIdx ] = true;
              bestMtsFlag               = mtsFlag;
              bestLfnstIdx              = lfnstIdx;
              if( bestCS->cus.size() == 1 )
              {
                CodingUnit &cu = *bestCS->cus.front();
                if (cu.firstTU->mtsIdx[COMPONENT_Y] == MTS_SKIP)
                {
                  if( ( floorLog2( cu.firstTU->blocks[ COMPONENT_Y ].width ) + floorLog2( cu.firstTU->blocks[ COMPONENT_Y ].height ) ) >= 6 )
                  {
                    endLfnstIdx = 0;
                  }
                }
              }
            }

            //we decide to skip the non-DCT-II transforms and LFNST according to the ISP results
            if ((endMtsFlag > 0 || endLfnstIdx > 0) && (cu.ispMode || (bestCS && bestCS->cus[0]->ispMode)) && tempCS->slice->isIntra() && m_pcEncCfg->getUseFastISP())
            {
              double bestCostDct2NoIsp = m_modeCtrl->getMtsFirstPassNoIspCost();
              double bestIspCost       = m_modeCtrl->getIspCost();
              CHECKD( bestCostDct2NoIsp <= bestIspCost, "wrong cost!" );
              double threshold = 1.4;

              double lfnstThreshold = 1.01 * threshold;
              if( m_modeCtrl->getStopNonDCT2Transforms() || bestCostDct2NoIsp > bestIspCost*lfnstThreshold )
              {
                endLfnstIdx = lfnstIdx;
              }

              if ( m_modeCtrl->getStopNonDCT2Transforms() || bestCostDct2NoIsp > bestIspCost*threshold )
              {
                skipSecondMtsPass = true;
                m_modeCtrl->setSkipSecondMTSPass( true );
                break;
              }
            }
            //now we check whether the second pass of SIZE_2Nx2N and the whole Intra SIZE_NxN should be skipped or not
            if( !mtsFlag && !tempCS->slice->isIntra() && bestCU && bestCU->predMode != MODE_INTRA )
            {
              const double thEmtInterFastSkipIntra = 1.4; // Skip checking Intra if "2Nx2N using DCT2" is worse than best Inter mode
              if( costSize2Nx2NmtsFirstPass > thEmtInterFastSkipIntra * bestInterCost )
              {
                skipSecondMtsPass = true;
                m_modeCtrl->setSkipSecondMTSPass( true );
                break;
              }
            }
          }

        } //for emtCuFlag
        if( skipOtherLfnst )
        {
          startLfnstIdx = lfnstIdx;
          endLfnstIdx   = lfnstIdx;
          break;
        }
      } //for lfnstIdx
    } //if (!skipSecondMtsPass && considerMtsSecondPass && trGrpCheck[iGrpIdx])

    if( sps.getUseLFNST() && trGrpIdx < 3 )
    {
      trGrpCheck[ trGrpIdx + 1 ] = false;

      if( bestSelFlag[ trGrpIdx ] && considerMtsSecondPass )
      {
        double dCostRatio = dct2Cost / trGrpBestCost[ trGrpIdx ];
        trGrpCheck[ trGrpIdx + 1 ] = ( bestMtsFlag != 0 || bestLfnstIdx != 0 ) && dCostRatio < trGrpStopThreshold[ trGrpIdx ];
      }
    }
  } //trGrpIdx
  if(!adaptiveColorTrans)
  m_modeCtrl->setBestNonDCT2Cost(bestNonDCT2Cost);
  return foundZeroRootCbf;
}


void EncCu::xCheckPLT(CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode& encTestMode)
{
  if (((partitioner.currArea().lumaSize().width * partitioner.currArea().lumaSize().height <= 16) && (isLuma(partitioner.chType)) )
        || ((partitioner.currArea().chromaSize().width * partitioner.currArea().chromaSize().height <= 16) && (!isLuma(partitioner.chType)) && partitioner.isSepTree(*tempCS) ) )
  {
    return;
  }
  tempCS->initStructData(encTestMode.qp);
  CodingUnit &cu = tempCS->addCU(CS::getArea(*tempCS, tempCS->area, partitioner.chType), partitioner.chType);
  partitioner.setCUData(cu);
  cu.slice = tempCS->slice;
  cu.tileIdx = tempCS->pps->getTileIdx(tempCS->area.lumaPos());
  cu.skip = false;
  cu.mmvdSkip = false;
  cu.predMode = MODE_PLT;

  cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
  cu.qp = encTestMode.qp;
  cu.bdpcmMode = 0;

  tempCS->addPU(CS::getArea(*tempCS, tempCS->area, partitioner.chType), partitioner.chType);
  tempCS->addTU(CS::getArea(*tempCS, tempCS->area, partitioner.chType), partitioner.chType);
  // Search
  tempCS->dist = 0;
  if (cu.isSepTree())
  {
    if (isLuma(partitioner.chType))
    {
      m_pcIntraSearch->PLTSearch(*tempCS, partitioner, COMPONENT_Y, 1);
    }
    if (tempCS->area.chromaFormat != CHROMA_400 && (partitioner.chType == CHANNEL_TYPE_CHROMA))
    {
      m_pcIntraSearch->PLTSearch(*tempCS, partitioner, COMPONENT_Cb, 2);
    }
  }
  else
  {
    if( cu.chromaFormat != CHROMA_400 )
    {
      m_pcIntraSearch->PLTSearch(*tempCS, partitioner, COMPONENT_Y, 3);
    }
    else
    {
      m_pcIntraSearch->PLTSearch(*tempCS, partitioner, COMPONENT_Y, 1);
    }
  }


  m_CABACEstimator->getCtx() = m_CurrCtx->start;
  m_CABACEstimator->resetBits();
  if ((!cu.cs->slice->isIntra() || cu.cs->slice->getSPS()->getIBCFlag())
    && cu.Y().valid())
  {
    m_CABACEstimator->cu_skip_flag(cu);
  }
  m_CABACEstimator->pred_mode(cu);

  // signaling
  CUCtx cuCtx;
  cuCtx.isDQPCoded = true;
  cuCtx.isChromaQpAdjCoded = true;
  if (cu.isSepTree())
  {
    if (isLuma(partitioner.chType))
    {
      m_CABACEstimator->cu_palette_info(cu, COMPONENT_Y, 1, cuCtx);
    }
    if (tempCS->area.chromaFormat != CHROMA_400 && (partitioner.chType == CHANNEL_TYPE_CHROMA))
    {
      m_CABACEstimator->cu_palette_info(cu, COMPONENT_Cb, 2, cuCtx);
    }
  }
  else
  {
    if( cu.chromaFormat != CHROMA_400 )
    {
      m_CABACEstimator->cu_palette_info(cu, COMPONENT_Y, 3, cuCtx);
    }
    else
    {
      m_CABACEstimator->cu_palette_info(cu, COMPONENT_Y, 1, cuCtx);
    }
  }
  tempCS->fracBits = m_CABACEstimator->getEstFracBits();
  tempCS->cost = m_pcRdCost->calcRdCost(tempCS->fracBits, tempCS->dist);

  xEncodeDontSplit(*tempCS, partitioner);
  xCheckDQP(*tempCS, partitioner);
  xCheckChromaQPOffset( *tempCS, partitioner );
  xCalDebCost(*tempCS, partitioner);
  tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();

  const Area currCuArea = cu.block(getFirstComponentOfChannel(partitioner.chType));
  cu.slice->m_mapPltCost[isChroma(partitioner.chType)][currCuArea.pos()][currCuArea.size()] = tempCS->cost;
#if WCG_EXT
  DTRACE_MODE_COST(*tempCS, m_pcRdCost->getLambda(true));
#else
  DTRACE_MODE_COST(*tempCS, m_pcRdCost->getLambda());
#endif
  xCheckBestMode(tempCS, bestCS, partitioner, encTestMode);
}

void EncCu::xCheckDQP( CodingStructure& cs, Partitioner& partitioner, bool bKeepCtx )
{
  CHECK_VTM( bKeepCtx && cs.cus.size() <= 1 && partitioner.getImplicitSplit( cs ) == CU_DONT_SPLIT, "bKeepCtx should only be set in split case" );
  CHECK_VTM( !bKeepCtx && cs.cus.size() > 1, "bKeepCtx should never be set for non-split case" );

  if( !cs.pps->getUseDQP() )
  {
    return;
  }

  if (partitioner.isSepTree(cs) && isChroma(partitioner.chType))
  {
    return;
  }

  if( !partitioner.currQgEnable() ) // do not consider split or leaf/not leaf QG condition (checked by caller)
  {
    return;
  }


  CodingUnit* cuFirst = cs.getCU( partitioner.chType );

  CHECK_VTM( !cuFirst, "No CU available" );

  bool hasResidual = false;
  for( const auto &cu : cs.cus )
  {
    //not include the chroma CU because chroma CU is decided based on corresponding luma QP and deltaQP is not signaled at chroma CU
    if( cu->rootCbf && !isChroma( cu->chType ))
    {
      hasResidual = true;
      break;
    }
  }

  int predQP = CU::predictQP( *cuFirst, cs.prevQP[partitioner.chType] );

  if( hasResidual )
  {
    TempCtx ctxTemp( m_CtxCache );
    if( !bKeepCtx ) ctxTemp = SubCtx( Ctx::DeltaQP, m_CABACEstimator->getCtx() );

    m_CABACEstimator->resetBits();
    m_CABACEstimator->cu_qp_delta( *cuFirst, predQP, cuFirst->qp );

    cs.fracBits += m_CABACEstimator->getEstFracBits(); // dQP bits
    cs.cost      = m_pcRdCost->calcRdCost(cs.fracBits, cs.dist);


    if( !bKeepCtx ) m_CABACEstimator->getCtx() = SubCtx( Ctx::DeltaQP, ctxTemp );

    // NOTE: reset QPs for CUs without residuals up to first coded CU
    for( const auto &cu : cs.cus )
    {
      //not include the chroma CU because chroma CU is decided based on corresponding luma QP and deltaQP is not signaled at chroma CU
      if( cu->rootCbf && !isChroma( cu->chType ))
      {
        break;
      }
      cu->qp = predQP;
    }
  }
  else
  {
    // No residuals: reset CU QP to predicted value
    for( const auto &cu : cs.cus )
    {
      cu->qp = predQP;
    }
  }
}

void EncCu::xCheckChromaQPOffset( CodingStructure& cs, Partitioner& partitioner )
{
  // doesn't apply if CU chroma QP offset is disabled
  if( !cs.slice->getUseChromaQpAdj() )
  {
    return;
  }

  // doesn't apply to luma CUs
  if( partitioner.isSepTree(cs) && isLuma(partitioner.chType) )
  {
    return;
  }

  // not needed after the first coded TU in the chroma QG
  if( !partitioner.currQgChromaEnable() )
  {
    return;
  }

  CodingUnit& cu = *cs.getCU( partitioner.chType );

  // check if chroma is coded or not
  bool hasResidual = false;
  for( const TransformUnit &tu : CU::traverseTUs(cu) )
  {
    if( tu.cbf[COMPONENT_Cb] || tu.cbf[COMPONENT_Cr] )
    {
      hasResidual = true;
      break;
    }
  }

  if( hasResidual )
  {
    // estimate cost for coding cu_chroma_qp_offset
    TempCtx ctxTempAdjFlag( m_CtxCache );
    TempCtx ctxTempAdjIdc( m_CtxCache );
    ctxTempAdjFlag = SubCtx( Ctx::ChromaQpAdjFlag, m_CABACEstimator->getCtx() );
    ctxTempAdjIdc = SubCtx( Ctx::ChromaQpAdjIdc,   m_CABACEstimator->getCtx() );
    m_CABACEstimator->resetBits();
    m_CABACEstimator->cu_chroma_qp_offset( cu );
    cs.fracBits += m_CABACEstimator->getEstFracBits();
    cs.cost      = m_pcRdCost->calcRdCost(cs.fracBits, cs.dist);
    m_CABACEstimator->getCtx() = SubCtx( Ctx::ChromaQpAdjFlag, ctxTempAdjFlag );
    m_CABACEstimator->getCtx() = SubCtx( Ctx::ChromaQpAdjIdc,  ctxTempAdjIdc  );
  }
  else
  {
    // reset chroma QP offset to 0 if it will not be coded
    cu.chromaQpAdj = 0;
  }
}

void EncCu::xFillPCMBuffer( CodingUnit &cu )
{
  const ChromaFormat format        = cu.chromaFormat;
  const uint32_t numberValidComponents = getNumberValidComponents(format);

  for( auto &tu : CU::traverseTUs( cu ) )
  {
    for( uint32_t ch = 0; ch < numberValidComponents; ch++ )
    {
      const ComponentID compID = ComponentID( ch );

      const CompArea &compArea = tu.blocks[ compID ];

      const CPelBuf source      = tu.cs->getOrgBuf( compArea );
             PelBuf destination = tu.getPcmbuf( compID );
      if (tu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag() && compID == COMPONENT_Y)
      {
        CompArea    tmpArea(COMPONENT_Y, compArea.chromaFormat, Position(0, 0), compArea.size());
        PelBuf tempOrgBuf = m_tmpStorageLCU->getBuf(tmpArea);
        tempOrgBuf.copyFrom(source);
        tempOrgBuf.rspSignal(m_pcReshape->getFwdLUT());
        destination.copyFrom(tempOrgBuf);
      }
      else
        destination.copyFrom( source );
    }
  }
}
void EncCu::xCheckRDCostHashInter( 
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode)
{
  bool isPerfectMatch = false;

  tempCS->initStructData(encTestMode.qp);
  m_pcInterSearch->resetBufferedUniMotions();
  m_pcInterSearch->setAffineModeSelected(false);
  CodingUnit &cu = tempCS->addCU(tempCS->area, partitioner.chType);

  partitioner.setCUData(cu);
  cu.slice = tempCS->slice;
  cu.tileIdx = tempCS->pps->getTileIdx(tempCS->area.lumaPos());
  cu.skip = false;
  cu.predMode = MODE_INTER;
  cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
  cu.qp = encTestMode.qp;
  CU::addPUs(cu);
  cu.mmvdSkip = false;
  cu.firstPU->mmvdMergeFlag = false;

  if (m_pcInterSearch->predInterHashSearch(cu, partitioner, isPerfectMatch))
  {
    double equBcwCost = MAX_DOUBLE;

    m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;

    xEncodeInterResidual(
#if ANNM
      flag,
#endif
      tempCS, bestCS, partitioner, encTestMode, 0
      , 0
      , &equBcwCost
    );

    if ( m_bestModeUpdated && bestCS->cost != MAX_DOUBLE )
    {
      xCalDebCost( *bestCS, partitioner );
    }
  }
  tempCS->initStructData(encTestMode.qp);
  int minSize = min(cu.lwidth(), cu.lheight());
  if (minSize < 64)
  {
    isPerfectMatch = false;
  }
  m_modeCtrl->setIsHashPerfectMatch(isPerfectMatch);
}

void EncCu::xCheckRDCostMerge2Nx2N( 
  #if ANNM
  bool& flag, Pel* img[], int pieX, int pieY,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode)
{
  const Slice &slice = *tempCS->slice;

  CHECK_VTM( slice.getSliceType() == I_SLICE, "Merge modes not available for I-slices" );

  tempCS->initStructData( encTestMode.qp );

  MergeCtx mergeCtx;
  const SPS &sps = *tempCS->sps;

  if( sps.getSBTMVPEnabledFlag() )
  {
    Size bufSize = g_miScaling.scale( tempCS->area.lumaSize() );
    mergeCtx.subPuMvpMiBuf    = MotionBuf( m_SubPuMiBuf,    bufSize );
  }

  Mv   refinedMvdL0[MAX_NUM_PARTS_IN_CTU][MRG_MAX_NUM_CANDS];
  setMergeBestSATDCost( MAX_DOUBLE );

  {
    // first get merge candidates
    CodingUnit cu( tempCS->area );
    cu.cs       = tempCS;
    cu.predMode = MODE_INTER;
    cu.slice    = tempCS->slice;
    cu.tileIdx  = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );

    PredictionUnit pu( tempCS->area );
    pu.cu = &cu;
    pu.cs = tempCS;
    PU::getInterMergeCandidates(pu, mergeCtx
      , 0
    );
    PU::getInterMMVDMergeCandidates(pu, mergeCtx);
    pu.regularMergeFlag = true;
  }
  bool candHasNoResidual[MRG_MAX_NUM_CANDS + MMVD_ADD_NUM];
  for (uint32_t ui = 0; ui < MRG_MAX_NUM_CANDS + MMVD_ADD_NUM; ui++)
  {
    candHasNoResidual[ui] = false;
  }

  bool                                        bestIsSkip = false;
  bool                                        bestIsMMVDSkip = true;
  PelUnitBuf                                  acMergeBuffer[MRG_MAX_NUM_CANDS];
  PelUnitBuf                                  acMergeTmpBuffer[MRG_MAX_NUM_CANDS];
  PelUnitBuf                                  acMergeRealBuffer[MMVD_MRG_MAX_RD_BUF_NUM];
  PelUnitBuf *                                acMergeTempBuffer[MMVD_MRG_MAX_RD_NUM];
  PelUnitBuf *                                singleMergeTempBuffer;
  int                                         insertPos;
  unsigned                                    uiNumMrgSATDCand = mergeCtx.numValidMergeCand + MMVD_ADD_NUM;

  struct ModeInfo
  {
    uint32_t mergeCand;
    bool     isRegularMerge;
    bool     isMMVD;
    bool     isCIIP;
    ModeInfo() : mergeCand(0), isRegularMerge(false), isMMVD(false), isCIIP(false) {}
    ModeInfo(const uint32_t mergeCand, const bool isRegularMerge, const bool isMMVD, const bool isCIIP) :
      mergeCand(mergeCand), isRegularMerge(isRegularMerge), isMMVD(isMMVD), isCIIP(isCIIP) {}
  };

  static_vector<ModeInfo, MRG_MAX_NUM_CANDS + MMVD_ADD_NUM>  RdModeList;
  bool                                        mrgTempBufSet = false;
  const int candNum = MRG_MAX_NUM_CANDS + (tempCS->sps->getUseMMVD() ? MMVD_ADD_NUM : 0);

  for (int i = 0; i < candNum; i++)
  {
    if (i < mergeCtx.numValidMergeCand)
    {
      RdModeList.push_back(ModeInfo(i, true, false, false));
    }
    else
    {
      RdModeList.push_back(ModeInfo(std::min(MMVD_ADD_NUM, i - mergeCtx.numValidMergeCand), false, true, false));
    }
  }

  const UnitArea localUnitArea(tempCS->area.chromaFormat, Area(0, 0, tempCS->area.Y().width, tempCS->area.Y().height));
  for (unsigned i = 0; i < MMVD_MRG_MAX_RD_BUF_NUM; i++)
  {
    acMergeRealBuffer[i] = m_acMergeBuffer[i].getBuf(localUnitArea);
    if (i < MMVD_MRG_MAX_RD_NUM)
    {
      acMergeTempBuffer[i] = acMergeRealBuffer + i;
    }
    else
    {
      singleMergeTempBuffer = acMergeRealBuffer + i;
    }
  }

  bool isIntrainterEnabled = sps.getUseCiip();
  if (bestCS->area.lwidth() * bestCS->area.lheight() < 64 || bestCS->area.lwidth() >= MAX_CU_SIZE || bestCS->area.lheight() >= MAX_CU_SIZE)
  {
    isIntrainterEnabled = false;
  }
  bool isTestSkipMerge[MRG_MAX_NUM_CANDS]; // record if the merge candidate has tried skip mode
  for (uint32_t idx = 0; idx < MRG_MAX_NUM_CANDS; idx++)
  {
    isTestSkipMerge[idx] = false;
  }
  if( m_pcEncCfg->getUseFastMerge() || isIntrainterEnabled)
  {
    uiNumMrgSATDCand = NUM_MRG_SATD_CAND;
    if (isIntrainterEnabled)
    {
      uiNumMrgSATDCand += 1;
    }
    bestIsSkip       = false;

    if( auto blkCache = dynamic_cast< CacheBlkInfoCtrl* >( m_modeCtrl ) )
    {
      if (slice.getSPS()->getIBCFlag())
      {
        ComprCUCtx cuECtx = m_modeCtrl->getComprCUCtx();
        bestIsSkip = blkCache->isSkip(tempCS->area) && cuECtx.bestCU;
      }
      else
      bestIsSkip = blkCache->isSkip( tempCS->area );
      bestIsMMVDSkip = blkCache->isMMVDSkip(tempCS->area);
    }

    if (isIntrainterEnabled) // always perform low complexity check
    {
      bestIsSkip = false;
    }

    static_vector<double, MRG_MAX_NUM_CANDS + MMVD_ADD_NUM> candCostList;

    // 1. Pass: get SATD-cost for selected candidates and reduce their count
    if( !bestIsSkip )
    {
      RdModeList.clear();
      mrgTempBufSet       = true;
      const TempCtx ctxStart(m_CtxCache, m_CABACEstimator->getCtx());

      CodingUnit &cu      = tempCS->addCU( tempCS->area, partitioner.chType );
      const double sqrtLambdaForFirstPassIntra = m_pcRdCost->getMotionLambda( ) * FRAC_BITS_SCALE;
      partitioner.setCUData( cu );
      cu.slice            = tempCS->slice;
      cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
      cu.skip             = false;
      cu.mmvdSkip = false;
      cu.geoFlag          = false;
    //cu.affine
      cu.predMode         = MODE_INTER;
    //cu.LICFlag
      cu.chromaQpAdj      = m_cuChromaQpOffsetIdxPlus1;
      cu.qp               = encTestMode.qp;
    //cu.emtFlag  is set below

      PredictionUnit &pu  = tempCS->addPU( cu, partitioner.chType );

      DistParam distParam;
      const bool bUseHadamard = !tempCS->slice->getDisableSATDForRD();
      m_pcRdCost->setDistParam (distParam, tempCS->getOrgBuf().Y(), m_acMergeBuffer[0].Y(), sps.getBitDepth (CHANNEL_TYPE_LUMA), COMPONENT_Y, bUseHadamard);

      const UnitArea localUnitArea( tempCS->area.chromaFormat, Area( 0, 0, tempCS->area.Y().width, tempCS->area.Y().height) );
      for( uint32_t uiMergeCand = 0; uiMergeCand < mergeCtx.numValidMergeCand; uiMergeCand++ )
      {
        mergeCtx.setMergeInfo( pu, uiMergeCand );

        PU::spanMotionInfo( pu, mergeCtx );
        pu.mvRefine = true;
        distParam.cur = singleMergeTempBuffer->Y();
        acMergeTmpBuffer[uiMergeCand] = m_acMergeTmpBuffer[uiMergeCand].getBuf(localUnitArea);
        m_pcInterSearch->motionCompensation(pu, *singleMergeTempBuffer, REF_PIC_LIST_X, true, true, &(acMergeTmpBuffer[uiMergeCand]));
        acMergeBuffer[uiMergeCand] = m_acRealMergeBuffer[uiMergeCand].getBuf(localUnitArea);
        acMergeBuffer[uiMergeCand].copyFrom(*singleMergeTempBuffer);
        pu.mvRefine = false;
        if( mergeCtx.interDirNeighbours[uiMergeCand] == 3 && mergeCtx.mrgTypeNeighbours[uiMergeCand] == MRG_TYPE_DEFAULT_N )
        {
          mergeCtx.mvFieldNeighbours[2*uiMergeCand].mv   = pu.mv[0];
          mergeCtx.mvFieldNeighbours[2*uiMergeCand+1].mv = pu.mv[1];
          {
            int dx, dy, i, j, num = 0;
            dy = std::min<int>(pu.lumaSize().height, DMVR_SUBCU_HEIGHT);
            dx = std::min<int>(pu.lumaSize().width, DMVR_SUBCU_WIDTH);
            if (PU::checkDMVRCondition(pu))
            {
              for (i = 0; i < (pu.lumaSize().height); i += dy)
              {
                for (j = 0; j < (pu.lumaSize().width); j += dx)
                {
                  refinedMvdL0[num][uiMergeCand] = pu.mvdL0SubPu[num];
                  num++;
                }
              }
            }
          }
        }

        Distortion uiSad = distParam.distFunc(distParam);
        m_CABACEstimator->getCtx() = ctxStart;
        uint64_t fracBits = m_pcInterSearch->xCalcPuMeBits(pu);
        double cost = (double)uiSad + (double)fracBits * sqrtLambdaForFirstPassIntra;
        insertPos = -1;
        updateCandList(ModeInfo(uiMergeCand, true, false, false), cost, RdModeList, candCostList, uiNumMrgSATDCand, &insertPos);
        if (insertPos != -1)
        {
          if (insertPos == RdModeList.size() - 1)
          {
            swap(singleMergeTempBuffer, acMergeTempBuffer[insertPos]);
          }
          else
          {
            for (uint32_t i = uint32_t(RdModeList.size()) - 1; i > insertPos; i--)
            {
              swap(acMergeTempBuffer[i - 1], acMergeTempBuffer[i]);
            }
            swap(singleMergeTempBuffer, acMergeTempBuffer[insertPos]);
          }
        }
        CHECK_VTM(std::min(uiMergeCand + 1, uiNumMrgSATDCand) != RdModeList.size(), "");
      }

      if (isIntrainterEnabled)
      {
        // prepare for Intra bits calculation
        pu.ciipFlag = true;

        // save the to-be-tested merge candidates
        uint32_t CiipMergeCand[NUM_MRG_SATD_CAND];
        for (uint32_t mergeCnt = 0; mergeCnt < std::min(NUM_MRG_SATD_CAND, (const int)mergeCtx.numValidMergeCand); mergeCnt++)
        {
          CiipMergeCand[mergeCnt] = RdModeList[mergeCnt].mergeCand;
        }
        for (uint32_t mergeCnt = 0; mergeCnt < std::min(std::min(NUM_MRG_SATD_CAND, (const int)mergeCtx.numValidMergeCand), 4); mergeCnt++)
        {
          uint32_t mergeCand = CiipMergeCand[mergeCnt];
          acMergeTmpBuffer[mergeCand] = m_acMergeTmpBuffer[mergeCand].getBuf(localUnitArea);

          // estimate merge bits
          mergeCtx.setMergeInfo(pu, mergeCand);

          // first round
          pu.intraDir[0] = PLANAR_IDX;
          uint32_t intraCnt = 0;
          // generate intrainter Y prediction
          if (mergeCnt == 0)
          {
            m_pcIntraSearch->initIntraPatternChType(*pu.cu, pu.Y());
            m_pcIntraSearch->predIntraAng(
#if ANNM
                img, pieX, pieY,
#endif
              COMPONENT_Y, pu.cs->getPredBuf(pu).Y(), pu);
            m_pcIntraSearch->switchBuffer(pu, COMPONENT_Y, pu.cs->getPredBuf(pu).Y(), m_pcIntraSearch->getPredictorPtr2(COMPONENT_Y, intraCnt));
          }
          pu.cs->getPredBuf(pu).copyFrom(acMergeTmpBuffer[mergeCand]);
          if (pu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
          {
            pu.cs->getPredBuf(pu).Y().rspSignal(m_pcReshape->getFwdLUT());
          }
          m_pcIntraSearch->geneWeightedPred(COMPONENT_Y, pu.cs->getPredBuf(pu).Y(), pu, m_pcIntraSearch->getPredictorPtr2(COMPONENT_Y, intraCnt));

          // calculate cost
          if (pu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
          {
            pu.cs->getPredBuf(pu).Y().rspSignal(m_pcReshape->getInvLUT());
          }
          distParam.cur = pu.cs->getPredBuf(pu).Y();
          Distortion sadValue = distParam.distFunc(distParam);
          if (pu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
          {
            pu.cs->getPredBuf(pu).Y().rspSignal(m_pcReshape->getFwdLUT());
          }
          m_CABACEstimator->getCtx() = ctxStart;
          pu.regularMergeFlag = false;
          uint64_t fracBits = m_pcInterSearch->xCalcPuMeBits(pu);
          double cost = (double)sadValue + (double)fracBits * sqrtLambdaForFirstPassIntra;
          insertPos = -1;
          updateCandList(ModeInfo(mergeCand, false, false, true), cost, RdModeList, candCostList, uiNumMrgSATDCand, &insertPos);
          if (insertPos != -1)
          {
            for (int i = int(RdModeList.size()) - 1; i > insertPos; i--)
            {
              swap(acMergeTempBuffer[i - 1], acMergeTempBuffer[i]);
            }
            swap(singleMergeTempBuffer, acMergeTempBuffer[insertPos]);
          }
        }
        pu.ciipFlag = false;
      }
      if ( pu.cs->sps->getUseMMVD() )
      {
        cu.mmvdSkip = true;
        pu.regularMergeFlag = true;
        const int tempNum = (mergeCtx.numValidMergeCand > 1) ? MMVD_ADD_NUM : MMVD_ADD_NUM >> 1;
        for (int mmvdMergeCand = 0; mmvdMergeCand < tempNum; mmvdMergeCand++)
        {
          int baseIdx = mmvdMergeCand / MMVD_MAX_REFINE_NUM;
          int refineStep = (mmvdMergeCand - (baseIdx * MMVD_MAX_REFINE_NUM)) / 4;
          if (refineStep >= m_pcEncCfg->getMmvdDisNum())
            continue;
          mergeCtx.setMmvdMergeCandiInfo(pu, mmvdMergeCand);

          PU::spanMotionInfo(pu, mergeCtx);
          pu.mvRefine = true;
          distParam.cur = singleMergeTempBuffer->Y();
          pu.mmvdEncOptMode = (refineStep > 2 ? 2 : 1);
          CHECK_VTM(!pu.mmvdMergeFlag, "MMVD merge should be set");
          // Don't do chroma MC here
          m_pcInterSearch->motionCompensation(pu, *singleMergeTempBuffer, REF_PIC_LIST_X, true, false);
          pu.mmvdEncOptMode = 0;
          pu.mvRefine = false;
          Distortion uiSad = distParam.distFunc(distParam);

          m_CABACEstimator->getCtx() = ctxStart;
          uint64_t fracBits = m_pcInterSearch->xCalcPuMeBits(pu);
          double cost = (double)uiSad + (double)fracBits * sqrtLambdaForFirstPassIntra;
          insertPos = -1;
          updateCandList(ModeInfo(mmvdMergeCand, false, true, false), cost, RdModeList, candCostList, uiNumMrgSATDCand, &insertPos);
          if (insertPos != -1)
          {
            for (int i = int(RdModeList.size()) - 1; i > insertPos; i--)
            {
              swap(acMergeTempBuffer[i - 1], acMergeTempBuffer[i]);
            }
            swap(singleMergeTempBuffer, acMergeTempBuffer[insertPos]);
          }
        }
      }
      // Try to limit number of candidates using SATD-costs
      for( uint32_t i = 1; i < uiNumMrgSATDCand; i++ )
      {
        if( candCostList[i] > MRG_FAST_RATIO * candCostList[0] )
        {
          uiNumMrgSATDCand = i;
          break;
        }
      }

      setMergeBestSATDCost( candCostList[0] );

      if (isIntrainterEnabled && isChromaEnabled(pu.cs->pcv->chrFormat))
      {
        pu.ciipFlag = true;
        for (uint32_t mergeCnt = 0; mergeCnt < uiNumMrgSATDCand; mergeCnt++)
        {
          if (RdModeList[mergeCnt].isCIIP)
          {
            pu.intraDir[0] = PLANAR_IDX;
            pu.intraDir[1] = DM_CHROMA_IDX;
            if (pu.chromaSize().width == 2)
              continue;
            uint32_t bufIdx = 0;
            m_pcIntraSearch->initIntraPatternChType(*pu.cu, pu.Cb());
            m_pcIntraSearch->predIntraAng(
#if ANNM
                img, pieX, pieY,
#endif
              
              COMPONENT_Cb, pu.cs->getPredBuf(pu).Cb(), pu);
            m_pcIntraSearch->switchBuffer(pu, COMPONENT_Cb, pu.cs->getPredBuf(pu).Cb(), m_pcIntraSearch->getPredictorPtr2(COMPONENT_Cb, bufIdx));

            m_pcIntraSearch->initIntraPatternChType(*pu.cu, pu.Cr());
            m_pcIntraSearch->predIntraAng(
#if ANNM
                img, pieX, pieY,
#endif
              COMPONENT_Cr, pu.cs->getPredBuf(pu).Cr(), pu);
            m_pcIntraSearch->switchBuffer(pu, COMPONENT_Cr, pu.cs->getPredBuf(pu).Cr(), m_pcIntraSearch->getPredictorPtr2(COMPONENT_Cr, bufIdx));
          }
        }
        pu.ciipFlag = false;
      }

      tempCS->initStructData( encTestMode.qp );
      m_CABACEstimator->getCtx() = ctxStart;
    }
    else
    {
      if (bestIsMMVDSkip)
      {
        uiNumMrgSATDCand = mergeCtx.numValidMergeCand + ((mergeCtx.numValidMergeCand > 1) ? MMVD_ADD_NUM : MMVD_ADD_NUM >> 1);
      }
      else
      {
        uiNumMrgSATDCand = mergeCtx.numValidMergeCand;
      }
    }
  }
  m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;
  uint32_t iteration;
  uint32_t iterationBegin = 0;
  iteration = 2;
  for (uint32_t uiNoResidualPass = iterationBegin; uiNoResidualPass < iteration; ++uiNoResidualPass)
  {
    for( uint32_t uiMrgHADIdx = 0; uiMrgHADIdx < uiNumMrgSATDCand; uiMrgHADIdx++ )
    {
      uint32_t uiMergeCand = RdModeList[uiMrgHADIdx].mergeCand;

      if (uiNoResidualPass != 0 && RdModeList[uiMrgHADIdx].isCIIP) // intrainter does not support skip mode
      {
        if (isTestSkipMerge[uiMergeCand])
        {
          continue;
        }
      }

      if (((uiNoResidualPass != 0) && candHasNoResidual[uiMrgHADIdx])
       || ( (uiNoResidualPass == 0) && bestIsSkip ) )
      {
        continue;
      }

      // first get merge candidates
      CodingUnit &cu      = tempCS->addCU( tempCS->area, partitioner.chType );

      partitioner.setCUData( cu );
      cu.slice            = tempCS->slice;
      cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
      cu.skip             = false;
      cu.mmvdSkip = false;
      cu.geoFlag          = false;
    //cu.affine
      cu.predMode         = MODE_INTER;
    //cu.LICFlag
      cu.chromaQpAdj      = m_cuChromaQpOffsetIdxPlus1;
      cu.qp               = encTestMode.qp;
      PredictionUnit &pu  = tempCS->addPU( cu, partitioner.chType );

      if (uiNoResidualPass == 0 && RdModeList[uiMrgHADIdx].isCIIP)
      {
        cu.mmvdSkip = false;
        mergeCtx.setMergeInfo(pu, uiMergeCand);
        pu.ciipFlag = true;
        pu.regularMergeFlag = false;
        pu.intraDir[0] = PLANAR_IDX;
        CHECK_VTM(pu.intraDir[0]<0 || pu.intraDir[0]>(NUM_LUMA_MODE - 1), "out of intra mode");
        pu.intraDir[1] = DM_CHROMA_IDX;
      }
      else if (RdModeList[uiMrgHADIdx].isMMVD)
      {
        cu.mmvdSkip = true;
        pu.regularMergeFlag = true;
        mergeCtx.setMmvdMergeCandiInfo(pu, uiMergeCand);
      }
      else
      {
        cu.mmvdSkip = false;
        pu.regularMergeFlag = true;
        mergeCtx.setMergeInfo(pu, uiMergeCand);
      }
      PU::spanMotionInfo( pu, mergeCtx );

      if( m_pcEncCfg->getMCTSEncConstraint() )
      {
        bool isDMVR = PU::checkDMVRCondition( pu );
        if( ( isDMVR && MCTSHelper::isRefBlockAtRestrictedTileBoundary( pu ) ) || ( !isDMVR && !( MCTSHelper::checkMvBufferForMCTSConstraint( pu ) ) ) )
        {
          // Do not use this mode
          tempCS->initStructData( encTestMode.qp );
          continue;
        }
      }
      if( mrgTempBufSet )
      {
        {
          int dx, dy, i, j, num = 0;
          dy = std::min<int>(pu.lumaSize().height, DMVR_SUBCU_HEIGHT);
          dx = std::min<int>(pu.lumaSize().width, DMVR_SUBCU_WIDTH);
          if (PU::checkDMVRCondition(pu))
          {
            for (i = 0; i < (pu.lumaSize().height); i += dy)
            {
              for (j = 0; j < (pu.lumaSize().width); j += dx)
              {
                pu.mvdL0SubPu[num] = refinedMvdL0[num][uiMergeCand];
                num++;
              }
            }
          }
        }
        if (pu.ciipFlag)
        {
          uint32_t bufIdx = 0;
          PelBuf tmpBuf = tempCS->getPredBuf(pu).Y();
          tmpBuf.copyFrom(acMergeTmpBuffer[uiMergeCand].Y());
          if (pu.cs->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
          {
            tmpBuf.rspSignal(m_pcReshape->getFwdLUT());
          }
          m_pcIntraSearch->geneWeightedPred(COMPONENT_Y, tmpBuf, pu, m_pcIntraSearch->getPredictorPtr2(COMPONENT_Y, bufIdx));
          if (isChromaEnabled(pu.chromaFormat))
          {
          if (pu.chromaSize().width > 2)
          {
          tmpBuf = tempCS->getPredBuf(pu).Cb();
          tmpBuf.copyFrom(acMergeTmpBuffer[uiMergeCand].Cb());
          m_pcIntraSearch->geneWeightedPred(COMPONENT_Cb, tmpBuf, pu, m_pcIntraSearch->getPredictorPtr2(COMPONENT_Cb, bufIdx));
          tmpBuf = tempCS->getPredBuf(pu).Cr();
          tmpBuf.copyFrom(acMergeTmpBuffer[uiMergeCand].Cr());
          m_pcIntraSearch->geneWeightedPred(COMPONENT_Cr, tmpBuf, pu, m_pcIntraSearch->getPredictorPtr2(COMPONENT_Cr, bufIdx));
          }
          else
          {
            tmpBuf = tempCS->getPredBuf(pu).Cb();
            tmpBuf.copyFrom(acMergeTmpBuffer[uiMergeCand].Cb());
            tmpBuf = tempCS->getPredBuf(pu).Cr();
            tmpBuf.copyFrom(acMergeTmpBuffer[uiMergeCand].Cr());
          }
          }
        }
        else
        {
          if (RdModeList[uiMrgHADIdx].isMMVD)
          {
            pu.mmvdEncOptMode = 0;
            m_pcInterSearch->motionCompensation(pu);
          }
          else if (uiNoResidualPass != 0 && RdModeList[uiMrgHADIdx].isCIIP)
          {
            tempCS->getPredBuf().copyFrom(acMergeBuffer[uiMergeCand]);
          }
          else
          {
            tempCS->getPredBuf().copyFrom(*acMergeTempBuffer[uiMrgHADIdx]);
          }
        }
      }
      else
      {
        pu.mvRefine = true;
        m_pcInterSearch->motionCompensation( pu );
        pu.mvRefine = false;
      }
      if (!cu.mmvdSkip && !pu.ciipFlag && uiNoResidualPass != 0)
      {
        CHECK_VTM(uiMergeCand >= mergeCtx.numValidMergeCand, "out of normal merge");
        isTestSkipMerge[uiMergeCand] = true;
      }

      xEncodeInterResidual( 
#if ANNM
        flag,
#endif
        tempCS, bestCS, partitioner, encTestMode, uiNoResidualPass,                                                      \
  uiNoResidualPass == 0 ? &candHasNoResidual[uiMrgHADIdx] : NULL);

      if( m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip && !pu.ciipFlag)
      {
        bestIsSkip = !bestCS->cus.empty() && bestCS->getCU( partitioner.chType )->rootCbf == 0;
      }
      tempCS->initStructData( encTestMode.qp );
    }// end loop uiMrgHADIdx

    if( uiNoResidualPass == 0 && m_pcEncCfg->getUseEarlySkipDetection() )
    {
      const CodingUnit     &bestCU = *bestCS->getCU( partitioner.chType );
      const PredictionUnit &bestPU = *bestCS->getPU( partitioner.chType );

      if( bestCU.rootCbf == 0 )
      {
        if( bestPU.mergeFlag )
        {
          m_modeCtrl->setEarlySkipDetected();
        }
        else if( m_pcEncCfg->getMotionEstimationSearchMethod() != MESEARCH_SELECTIVE )
        {
          int absolute_MV = 0;

          for( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            if( slice.getNumRefIdx( RefPicList( uiRefListIdx ) ) > 0 )
            {
              absolute_MV += bestPU.mvd[uiRefListIdx].getAbsHor() + bestPU.mvd[uiRefListIdx].getAbsVer();
            }
          }

          if( absolute_MV == 0 )
          {
            m_modeCtrl->setEarlySkipDetected();
          }
        }
      }
    }
  }
  if ( m_bestModeUpdated && bestCS->cost != MAX_DOUBLE )
  {
    xCalDebCost( *bestCS, partitioner );
  }
}

void EncCu::xCheckRDCostMergeGeo2Nx2N(
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &pm, const EncTestMode &encTestMode)
{
  const Slice &slice = *tempCS->slice;
  CHECK_VTM(slice.getSliceType() == I_SLICE, "Merge modes not available for I-slices");

  tempCS->initStructData(encTestMode.qp);

  MergeCtx mergeCtx;
  const SPS &sps = *tempCS->sps;

  if (sps.getSBTMVPEnabledFlag())
  {
    Size bufSize = g_miScaling.scale(tempCS->area.lumaSize());
    mergeCtx.subPuMvpMiBuf = MotionBuf(m_SubPuMiBuf, bufSize);
  }

  CodingUnit &cu = tempCS->addCU(tempCS->area, pm.chType);
  pm.setCUData(cu);
  cu.predMode = MODE_INTER;
  cu.slice = tempCS->slice;
  cu.tileIdx = tempCS->pps->getTileIdx(tempCS->area.lumaPos());
  cu.qp = encTestMode.qp;
  cu.affine = false;
  cu.mtsFlag = false;
  cu.BcwIdx = BCW_DEFAULT;
  cu.geoFlag = true;
  cu.imv = 0;
  cu.mmvdSkip = false;
  cu.skip = false;
  cu.mipFlag = false;
  cu.bdpcmMode = 0;

  PredictionUnit &pu = tempCS->addPU(cu, pm.chType);
  pu.mergeFlag = true;
  pu.regularMergeFlag = false;
  PU::getGeoMergeCandidates(pu, mergeCtx);

  GeoComboCostList comboList;
  int bitsCandTB = floorLog2(GEO_NUM_PARTITION_MODE);
  PelUnitBuf geoBuffer[MRG_MAX_NUM_CANDS];
  PelUnitBuf geoTempBuf[MRG_MAX_NUM_CANDS];
  PelUnitBuf geoCombinations[GEO_MAX_TRY_WEIGHTED_SAD];
  DistParam distParam;

  const UnitArea localUnitArea(tempCS->area.chromaFormat, Area(0, 0, tempCS->area.Y().width, tempCS->area.Y().height));
  const double sqrtLambdaForFirstPass = m_pcRdCost->getMotionLambda();
  uint8_t maxNumMergeCandidates = cu.cs->sps->getMaxNumGeoCand();
  DistParam distParamWholeBlk;
  m_pcRdCost->setDistParam(distParamWholeBlk, tempCS->getOrgBuf().Y(), m_acMergeBuffer[0].Y().buf, m_acMergeBuffer[0].Y().stride, sps.getBitDepth(CHANNEL_TYPE_LUMA), COMPONENT_Y);
  Distortion bestWholeBlkSad = MAX_UINT64;
  double bestWholeBlkCost = MAX_DOUBLE;
  Distortion *sadWholeBlk;
  sadWholeBlk = new Distortion[maxNumMergeCandidates];
  int *pocMrg;
  Mv *MrgMv;
  bool *isSkipThisCand;
  pocMrg = new int[maxNumMergeCandidates];
  MrgMv = new Mv[maxNumMergeCandidates];
  isSkipThisCand = new bool[maxNumMergeCandidates];
  for (int i = 0; i < maxNumMergeCandidates; i++)
    isSkipThisCand[i] = false;
  for (uint8_t mergeCand = 0; mergeCand < maxNumMergeCandidates; mergeCand++)
  {
    geoBuffer[mergeCand] = m_acMergeBuffer[mergeCand].getBuf(localUnitArea);
    mergeCtx.setMergeInfo(pu, mergeCand);
    int MrgList = mergeCtx.mvFieldNeighbours[(mergeCand << 1) + 0].refIdx == -1 ? 1 : 0;
    RefPicList MrgeRefPicList = (MrgList ? REF_PIC_LIST_1 : REF_PIC_LIST_0);
    int MrgrefIdx = mergeCtx.mvFieldNeighbours[(mergeCand << 1) + MrgList].refIdx;
    pocMrg[mergeCand] = tempCS->slice->getRefPic(MrgeRefPicList, MrgrefIdx)->getPOC();
    MrgMv[mergeCand] = mergeCtx.mvFieldNeighbours[(mergeCand << 1) + MrgList].mv;
    if (mergeCand)
    {
      for (int i = 0; i < mergeCand; i++)
      {
        if (pocMrg[mergeCand] == pocMrg[i] && MrgMv[mergeCand] == MrgMv[i])
        {
          isSkipThisCand[mergeCand] = true;
          break;
        }
      }
    }
    PU::spanMotionInfo(pu, mergeCtx);
    if (m_pcEncCfg->getMCTSEncConstraint() && (!(MCTSHelper::checkMvBufferForMCTSConstraint(pu))))
    {
      tempCS->initStructData(encTestMode.qp);
      return;
    }
    m_pcInterSearch->motionCompensation(pu, geoBuffer[mergeCand]);
    geoTempBuf[mergeCand] = m_acMergeTmpBuffer[mergeCand].getBuf(localUnitArea);
    geoTempBuf[mergeCand].Y().copyFrom(geoBuffer[mergeCand].Y());
    geoTempBuf[mergeCand].Y().roundToOutputBitdepth(geoTempBuf[mergeCand].Y(), cu.slice->clpRng(COMPONENT_Y));
    distParamWholeBlk.cur.buf = geoTempBuf[mergeCand].Y().buf;
    distParamWholeBlk.cur.stride = geoTempBuf[mergeCand].Y().stride;
    sadWholeBlk[mergeCand] = distParamWholeBlk.distFunc(distParamWholeBlk);
    if (sadWholeBlk[mergeCand] < bestWholeBlkSad)
    {
      bestWholeBlkSad = sadWholeBlk[mergeCand];
      int bitsCand = mergeCand + 1;
      bestWholeBlkCost = (double)bestWholeBlkSad + (double)bitsCand * sqrtLambdaForFirstPass;
    }
  }
  bool isGeo = true;
  for (uint8_t mergeCand = 1; mergeCand < maxNumMergeCandidates; mergeCand++)
  {
    isGeo &= isSkipThisCand[mergeCand];
  }
  if (isGeo)
  {
    return;
  }

  int wIdx = floorLog2(cu.lwidth()) - GEO_MIN_CU_LOG2;
  int hIdx = floorLog2(cu.lheight()) - GEO_MIN_CU_LOG2;
  for (int splitDir = 0; splitDir < GEO_NUM_PARTITION_MODE; splitDir++)
  {
    int maskStride = 0, maskStride2 = 0;
    int stepX = 1;
    Pel* SADmask;
    int16_t angle = g_GeoParams[splitDir][0];
    if (g_angle2mirror[angle] == 2)
    {
      maskStride = -GEO_WEIGHT_MASK_SIZE;
      maskStride2 = -(int)cu.lwidth();
      SADmask = &g_globalGeoEncSADmask[g_angle2mask[g_GeoParams[splitDir][0]]][(GEO_WEIGHT_MASK_SIZE - 1 - g_weightOffset[splitDir][hIdx][wIdx][1]) * GEO_WEIGHT_MASK_SIZE + g_weightOffset[splitDir][hIdx][wIdx][0]];
    }
    else if (g_angle2mirror[angle] == 1)
    {
      stepX = -1;
      maskStride2 = cu.lwidth();
      maskStride = GEO_WEIGHT_MASK_SIZE;
      SADmask = &g_globalGeoEncSADmask[g_angle2mask[g_GeoParams[splitDir][0]]][g_weightOffset[splitDir][hIdx][wIdx][1] * GEO_WEIGHT_MASK_SIZE + (GEO_WEIGHT_MASK_SIZE - 1 - g_weightOffset[splitDir][hIdx][wIdx][0])];
    }
    else
    {
      maskStride = GEO_WEIGHT_MASK_SIZE;
      maskStride2 = -(int)cu.lwidth();
      SADmask = &g_globalGeoEncSADmask[g_angle2mask[g_GeoParams[splitDir][0]]][g_weightOffset[splitDir][hIdx][wIdx][1] * GEO_WEIGHT_MASK_SIZE + g_weightOffset[splitDir][hIdx][wIdx][0]];
    }
    Distortion sadSmall = 0, sadLarge = 0;
    for (uint8_t mergeCand = 0; mergeCand < maxNumMergeCandidates; mergeCand++)
    {
      int bitsCand = mergeCand + 1;

      m_pcRdCost->setDistParam(distParam, tempCS->getOrgBuf().Y(), geoTempBuf[mergeCand].Y().buf, geoTempBuf[mergeCand].Y().stride, SADmask, maskStride, stepX, maskStride2, sps.getBitDepth(CHANNEL_TYPE_LUMA), COMPONENT_Y);
      sadLarge = distParam.distFunc(distParam);
      m_GeoCostList.insert(splitDir, 0, mergeCand, (double)sadLarge + (double)bitsCand * sqrtLambdaForFirstPass);
      sadSmall = sadWholeBlk[mergeCand] - sadLarge;
      m_GeoCostList.insert(splitDir, 1, mergeCand, (double)sadSmall + (double)bitsCand * sqrtLambdaForFirstPass);
    }
  }
  delete[] sadWholeBlk;
  delete[] pocMrg;
  delete[] MrgMv;
  delete[] isSkipThisCand;

  for (int splitDir = 0; splitDir < GEO_NUM_PARTITION_MODE; splitDir++)
  {
    for (int GeoMotionIdx = 0; GeoMotionIdx < maxNumMergeCandidates * (maxNumMergeCandidates - 1); GeoMotionIdx++)
    {
      unsigned int mergeCand0 = m_GeoModeTest[GeoMotionIdx].m_candIdx0;
      unsigned int mergeCand1 = m_GeoModeTest[GeoMotionIdx].m_candIdx1;
      double tempCost = m_GeoCostList.singleDistList[0][splitDir][mergeCand0].cost + m_GeoCostList.singleDistList[1][splitDir][mergeCand1].cost;
      if (tempCost > bestWholeBlkCost)
        continue;
      tempCost = tempCost + (double)bitsCandTB * sqrtLambdaForFirstPass;
      comboList.list.push_back(GeoMergeCombo(splitDir, mergeCand0, mergeCand1, tempCost));
    }
  }
  if (comboList.list.empty())
    return;
  comboList.sortByCost();
  bool geocandHasNoResidual[GEO_MAX_TRY_WEIGHTED_SAD];
  for (int mergeCand = 0; mergeCand < GEO_MAX_TRY_WEIGHTED_SAD; mergeCand++)
  {
    geocandHasNoResidual[mergeCand] = false;
  }
  bool bestIsSkip = false;
  int geoNumCobo = (int)comboList.list.size();
  static_vector<uint8_t, GEO_MAX_TRY_WEIGHTED_SAD> geoRdModeList;
  static_vector<double, GEO_MAX_TRY_WEIGHTED_SAD> geocandCostList;

  DistParam distParamSAD2;
  const bool useHadamard = !tempCS->slice->getDisableSATDForRD();
  m_pcRdCost->setDistParam(distParamSAD2, tempCS->getOrgBuf().Y(), m_acMergeBuffer[0].Y(), sps.getBitDepth(CHANNEL_TYPE_LUMA), COMPONENT_Y, useHadamard);
  int geoNumMrgSATDCand = min(GEO_MAX_TRY_WEIGHTED_SATD, geoNumCobo);

  for (uint8_t candidateIdx = 0; candidateIdx < min(geoNumCobo, GEO_MAX_TRY_WEIGHTED_SAD); candidateIdx++)
  {
    int splitDir = comboList.list[candidateIdx].splitDir;
    int mergeCand0 = comboList.list[candidateIdx].mergeIdx0;
    int mergeCand1 = comboList.list[candidateIdx].mergeIdx1;

    geoCombinations[candidateIdx] = m_acGeoWeightedBuffer[candidateIdx].getBuf(localUnitArea);
    m_pcInterSearch->weightedGeoBlk(pu, splitDir, CHANNEL_TYPE_LUMA, geoCombinations[candidateIdx], geoBuffer[mergeCand0], geoBuffer[mergeCand1]);
    distParamSAD2.cur = geoCombinations[candidateIdx].Y();
    Distortion sad = distParamSAD2.distFunc(distParamSAD2);
    int mvBits = 2;
    mergeCand1 -= mergeCand1 < mergeCand0 ? 0 : 1;
    mvBits += mergeCand0;
    mvBits += mergeCand1;
    double updateCost = (double)sad + (double)(bitsCandTB + mvBits) * sqrtLambdaForFirstPass;
    comboList.list[candidateIdx].cost = updateCost;
    updateCandList(candidateIdx, updateCost, geoRdModeList, geocandCostList, geoNumMrgSATDCand);
  }
  for (uint8_t i = 0; i < geoNumMrgSATDCand; i++)
  {
    if (geocandCostList[i] > MRG_FAST_RATIO * geocandCostList[0] || geocandCostList[i] > getMergeBestSATDCost() || geocandCostList[i] > getAFFBestSATDCost())
    {
      geoNumMrgSATDCand = i;
      break;
    }
  }
  for (uint8_t i = 0; i < geoNumMrgSATDCand && isChromaEnabled(pu.chromaFormat); i++)
  {
    uint8_t candidateIdx = geoRdModeList[i];
    int splitDir = comboList.list[candidateIdx].splitDir;
    int mergeCand0 = comboList.list[candidateIdx].mergeIdx0;
    int mergeCand1 = comboList.list[candidateIdx].mergeIdx1;
    geoCombinations[candidateIdx] = m_acGeoWeightedBuffer[candidateIdx].getBuf(localUnitArea);
    m_pcInterSearch->weightedGeoBlk(pu, splitDir, CHANNEL_TYPE_CHROMA, geoCombinations[candidateIdx], geoBuffer[mergeCand0], geoBuffer[mergeCand1]);
  }

  m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;
  tempCS->initStructData(encTestMode.qp);
  uint8_t iteration;
  uint8_t iterationBegin = 0;
  iteration = 2;
  for (uint8_t noResidualPass = iterationBegin; noResidualPass < iteration; ++noResidualPass)
  {
    for (uint8_t mrgHADIdx = 0; mrgHADIdx < geoNumMrgSATDCand; mrgHADIdx++)
    {
      uint8_t candidateIdx = geoRdModeList[mrgHADIdx];
      if (((noResidualPass != 0) && geocandHasNoResidual[candidateIdx])
        || ((noResidualPass == 0) && bestIsSkip))
      {
        continue;
      }
      CodingUnit &cu = tempCS->addCU(tempCS->area, pm.chType);
      pm.setCUData(cu);
      cu.predMode = MODE_INTER;
      cu.slice = tempCS->slice;
      cu.tileIdx = tempCS->pps->getTileIdx(tempCS->area.lumaPos());
      cu.qp = encTestMode.qp;
      cu.affine = false;
      cu.mtsFlag = false;
      cu.BcwIdx = BCW_DEFAULT;
      cu.geoFlag = true;
      cu.imv = 0;
      cu.mmvdSkip = false;
      cu.skip = false;
      cu.mipFlag = false;
      cu.bdpcmMode = 0;
      PredictionUnit &pu = tempCS->addPU(cu, pm.chType);
      pu.mergeFlag = true;
      pu.regularMergeFlag = false;
      pu.geoSplitDir = comboList.list[candidateIdx].splitDir;
      pu.geoMergeIdx0 = comboList.list[candidateIdx].mergeIdx0;
      pu.geoMergeIdx1 = comboList.list[candidateIdx].mergeIdx1;
      pu.mmvdMergeFlag = false;
      pu.mmvdMergeIdx = MAX_UINT;

      PU::spanGeoMotionInfo(pu, mergeCtx, pu.geoSplitDir, pu.geoMergeIdx0, pu.geoMergeIdx1);
      tempCS->getPredBuf().copyFrom(geoCombinations[candidateIdx]);

      xEncodeInterResidual(
#if ANNM
          flag,
#endif
        tempCS, bestCS, pm, encTestMode, noResidualPass,                                                                 \
  (noResidualPass == 0 ? &geocandHasNoResidual[candidateIdx] : NULL));

      if (m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip)
      {
        bestIsSkip = bestCS->getCU(pm.chType)->rootCbf == 0;
      }
      tempCS->initStructData(encTestMode.qp);
    }
  }
  if (m_bestModeUpdated && bestCS->cost != MAX_DOUBLE)
  {
    xCalDebCost(*bestCS, pm);
  }
}

void EncCu::xCheckRDCostAffineMerge2Nx2N( 
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode)
{
  if( m_modeCtrl->getFastDeltaQp() )
  {
    return;
  }

  if ( bestCS->area.lumaSize().width < 8 || bestCS->area.lumaSize().height < 8 )
  {
    return;
  }
  m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;
  const Slice &slice = *tempCS->slice;

  CHECK_VTM( slice.getSliceType() == I_SLICE, "Affine Merge modes not available for I-slices" );

  tempCS->initStructData( encTestMode.qp );

  AffineMergeCtx affineMergeCtx;
  const SPS &sps = *tempCS->sps;

  setAFFBestSATDCost(MAX_DOUBLE);

  MergeCtx mrgCtx;
  if ( sps.getSBTMVPEnabledFlag() )
  {
    Size bufSize = g_miScaling.scale( tempCS->area.lumaSize() );
    mrgCtx.subPuMvpMiBuf = MotionBuf( m_SubPuMiBuf, bufSize );
    affineMergeCtx.mrgCtx = &mrgCtx;
  }

  {
    // first get merge candidates
    CodingUnit cu( tempCS->area );
    cu.cs = tempCS;
    cu.predMode = MODE_INTER;
    cu.slice = tempCS->slice;
    cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
    cu.mmvdSkip = false;

    PredictionUnit pu( tempCS->area );
    pu.cu = &cu;
    pu.cs = tempCS;
    pu.regularMergeFlag = false;
    PU::getAffineMergeCand( pu, affineMergeCtx );

    if ( affineMergeCtx.numValidMergeCand <= 0 )
    {
      return;
    }
  }

  bool candHasNoResidual[AFFINE_MRG_MAX_NUM_CANDS];
  for ( uint32_t ui = 0; ui < affineMergeCtx.numValidMergeCand; ui++ )
  {
    candHasNoResidual[ui] = false;
  }

  bool                                        bestIsSkip = false;
  uint32_t                                    uiNumMrgSATDCand = affineMergeCtx.numValidMergeCand;
  PelUnitBuf                                  acMergeBuffer[AFFINE_MRG_MAX_NUM_CANDS];
  static_vector<uint32_t, AFFINE_MRG_MAX_NUM_CANDS>  RdModeList;
  bool                                        mrgTempBufSet = false;

  for ( uint32_t i = 0; i < AFFINE_MRG_MAX_NUM_CANDS; i++ )
  {
    RdModeList.push_back( i );
  }

  if ( m_pcEncCfg->getUseFastMerge() )
  {
    uiNumMrgSATDCand = std::min( NUM_AFF_MRG_SATD_CAND, affineMergeCtx.numValidMergeCand );
    bestIsSkip = false;

    if ( auto blkCache = dynamic_cast<CacheBlkInfoCtrl*>(m_modeCtrl) )
    {
      bestIsSkip = blkCache->isSkip( tempCS->area );
    }

    static_vector<double, AFFINE_MRG_MAX_NUM_CANDS> candCostList;

    // 1. Pass: get SATD-cost for selected candidates and reduce their count
    if ( !bestIsSkip )
    {
      RdModeList.clear();
      mrgTempBufSet = true;
      const double sqrtLambdaForFirstPass = m_pcRdCost->getMotionLambda( );

      CodingUnit &cu = tempCS->addCU( tempCS->area, partitioner.chType );

      partitioner.setCUData( cu );
      cu.slice = tempCS->slice;
      cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
      cu.skip = false;
      cu.affine = true;
      cu.predMode = MODE_INTER;
      cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
      cu.qp = encTestMode.qp;

      PredictionUnit &pu = tempCS->addPU( cu, partitioner.chType );

      DistParam distParam;
      const bool bUseHadamard = !tempCS->slice->getDisableSATDForRD();
      m_pcRdCost->setDistParam( distParam, tempCS->getOrgBuf().Y(), m_acMergeBuffer[0].Y(), sps.getBitDepth( CHANNEL_TYPE_LUMA ), COMPONENT_Y, bUseHadamard );

      const UnitArea localUnitArea( tempCS->area.chromaFormat, Area( 0, 0, tempCS->area.Y().width, tempCS->area.Y().height ) );

      for ( uint32_t uiMergeCand = 0; uiMergeCand < affineMergeCtx.numValidMergeCand; uiMergeCand++ )
      {
        acMergeBuffer[uiMergeCand] = m_acMergeBuffer[uiMergeCand].getBuf( localUnitArea );

        // set merge information
        pu.interDir = affineMergeCtx.interDirNeighbours[uiMergeCand];
        pu.mergeFlag = true;
        pu.regularMergeFlag = false;
        pu.mergeIdx = uiMergeCand;
        cu.affineType = affineMergeCtx.affineType[uiMergeCand];
        cu.BcwIdx = affineMergeCtx.BcwIdx[uiMergeCand];

        pu.mergeType = affineMergeCtx.mergeType[uiMergeCand];
        if ( pu.mergeType == MRG_TYPE_SUBPU_ATMVP )
        {
          pu.refIdx[0] = affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 0][0].refIdx;
          pu.refIdx[1] = affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 1][0].refIdx;
          PU::spanMotionInfo( pu, mrgCtx );
        }
        else
        {
          PU::setAllAffineMvField( pu, affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 0], REF_PIC_LIST_0 );
          PU::setAllAffineMvField( pu, affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 1], REF_PIC_LIST_1 );

          PU::spanMotionInfo( pu );
        }

        distParam.cur = acMergeBuffer[uiMergeCand].Y();

        m_pcInterSearch->motionCompensation( pu, acMergeBuffer[uiMergeCand], REF_PIC_LIST_X, true, false );

        Distortion uiSad = distParam.distFunc( distParam );
        uint32_t   uiBitsCand = uiMergeCand + 1;
        if ( uiMergeCand == tempCS->picHeader->getMaxNumAffineMergeCand() - 1 )
        {
          uiBitsCand--;
        }
        double cost = (double)uiSad + (double)uiBitsCand * sqrtLambdaForFirstPass;
        updateCandList( uiMergeCand, cost, RdModeList, candCostList
          , uiNumMrgSATDCand );

        CHECK_VTM( std::min( uiMergeCand + 1, uiNumMrgSATDCand ) != RdModeList.size(), "" );
      }

      // Try to limit number of candidates using SATD-costs
      for ( uint32_t i = 1; i < uiNumMrgSATDCand; i++ )
      {
        if ( candCostList[i] > MRG_FAST_RATIO * candCostList[0] )
        {
          uiNumMrgSATDCand = i;
          break;
        }
      }

      tempCS->initStructData( encTestMode.qp );
      setAFFBestSATDCost(candCostList[0]);

    }
    else
    {
      uiNumMrgSATDCand = affineMergeCtx.numValidMergeCand;
    }
  }

  uint32_t iteration;
  uint32_t iterationBegin = 0;
  iteration = 2;
  for (uint32_t uiNoResidualPass = iterationBegin; uiNoResidualPass < iteration; ++uiNoResidualPass)
  {
    for ( uint32_t uiMrgHADIdx = 0; uiMrgHADIdx < uiNumMrgSATDCand; uiMrgHADIdx++ )
    {
      uint32_t uiMergeCand = RdModeList[uiMrgHADIdx];

      if ( ((uiNoResidualPass != 0) && candHasNoResidual[uiMergeCand])
        || ((uiNoResidualPass == 0) && bestIsSkip) )
      {
        continue;
      }

      // first get merge candidates
      CodingUnit &cu = tempCS->addCU( tempCS->area, partitioner.chType );

      partitioner.setCUData( cu );
      cu.slice = tempCS->slice;
      cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
      cu.skip = false;
      cu.affine = true;
      cu.predMode = MODE_INTER;
      cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
      cu.qp = encTestMode.qp;
      PredictionUnit &pu = tempCS->addPU( cu, partitioner.chType );

      // set merge information
      pu.mergeFlag = true;
      pu.mergeIdx = uiMergeCand;
      pu.interDir = affineMergeCtx.interDirNeighbours[uiMergeCand];
      cu.affineType = affineMergeCtx.affineType[uiMergeCand];
      cu.BcwIdx = affineMergeCtx.BcwIdx[uiMergeCand];

      pu.mergeType = affineMergeCtx.mergeType[uiMergeCand];
      if ( pu.mergeType == MRG_TYPE_SUBPU_ATMVP )
      {
        pu.refIdx[0] = affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 0][0].refIdx;
        pu.refIdx[1] = affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 1][0].refIdx;
        PU::spanMotionInfo( pu, mrgCtx );
      }
      else
      {
        PU::setAllAffineMvField( pu, affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 0], REF_PIC_LIST_0 );
        PU::setAllAffineMvField( pu, affineMergeCtx.mvFieldNeighbours[(uiMergeCand << 1) + 1], REF_PIC_LIST_1 );

        PU::spanMotionInfo( pu );
      }

      if( m_pcEncCfg->getMCTSEncConstraint() && ( !( MCTSHelper::checkMvBufferForMCTSConstraint( *cu.firstPU ) ) ) )
      {
        // Do not use this mode
        tempCS->initStructData( encTestMode.qp );
        return;
      }
      if ( mrgTempBufSet )
      {
        tempCS->getPredBuf().copyFrom(acMergeBuffer[uiMergeCand], true, false);   // Copy Luma Only
        m_pcInterSearch->motionCompensation(pu, REF_PIC_LIST_X, false, true);
      }
      else
      {
        m_pcInterSearch->motionCompensation( pu );
      }

      xEncodeInterResidual(
#if ANNM
        flag,
#endif
        tempCS, bestCS, partitioner, encTestMode, uiNoResidualPass,                                                      \
  (uiNoResidualPass == 0 ? &candHasNoResidual[uiMergeCand] : NULL));

      if ( m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip )
      {
        bestIsSkip = bestCS->getCU( partitioner.chType )->rootCbf == 0;
      }
      tempCS->initStructData( encTestMode.qp );
    }// end loop uiMrgHADIdx

    if ( uiNoResidualPass == 0 && m_pcEncCfg->getUseEarlySkipDetection() )
    {
      const CodingUnit     &bestCU = *bestCS->getCU( partitioner.chType );
      const PredictionUnit &bestPU = *bestCS->getPU( partitioner.chType );

      if ( bestCU.rootCbf == 0 )
      {
        if ( bestPU.mergeFlag )
        {
          m_modeCtrl->setEarlySkipDetected();
        }
        else if ( m_pcEncCfg->getMotionEstimationSearchMethod() != MESEARCH_SELECTIVE )
        {
          int absolute_MV = 0;

          for ( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            if ( slice.getNumRefIdx( RefPicList( uiRefListIdx ) ) > 0 )
            {
              absolute_MV += bestPU.mvd[uiRefListIdx].getAbsHor() + bestPU.mvd[uiRefListIdx].getAbsVer();
            }
          }

          if ( absolute_MV == 0 )
          {
            m_modeCtrl->setEarlySkipDetected();
          }
        }
      }
    }
  }
  if ( m_bestModeUpdated && bestCS->cost != MAX_DOUBLE )
  {
    xCalDebCost( *bestCS, partitioner );
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////
// ibc merge/skip mode check
void EncCu::xCheckRDCostIBCModeMerge2Nx2N(
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode)
{
  assert(partitioner.chType != CHANNEL_TYPE_CHROMA); // chroma IBC is derived
  if (tempCS->area.lwidth() == 128 || tempCS->area.lheight() == 128) // disable IBC mode larger than 64x64
  {
    return;
  }
  const SPS &sps = *tempCS->sps;

  tempCS->initStructData(encTestMode.qp);
  MergeCtx mergeCtx;


  if (sps.getSBTMVPEnabledFlag())
  {
    Size bufSize = g_miScaling.scale(tempCS->area.lumaSize());
    mergeCtx.subPuMvpMiBuf = MotionBuf(m_SubPuMiBuf, bufSize);
  }

  {
    // first get merge candidates
    CodingUnit cu(tempCS->area);
    cu.cs = tempCS;
    cu.predMode = MODE_IBC;
    cu.slice = tempCS->slice;
    cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
    PredictionUnit pu(tempCS->area);
    pu.cu = &cu;
    pu.cs = tempCS;
    cu.mmvdSkip = false;
    pu.mmvdMergeFlag = false;
    pu.regularMergeFlag = false;
    cu.geoFlag = false;
    PU::getIBCMergeCandidates(pu, mergeCtx);
  }

  int candHasNoResidual[MRG_MAX_NUM_CANDS];
  for (unsigned int ui = 0; ui < mergeCtx.numValidMergeCand; ui++)
  {
    candHasNoResidual[ui] = 0;
  }

  bool                                        bestIsSkip = false;
  unsigned                                    numMrgSATDCand = mergeCtx.numValidMergeCand;
  static_vector<unsigned, MRG_MAX_NUM_CANDS>  RdModeList(MRG_MAX_NUM_CANDS);
  for (unsigned i = 0; i < MRG_MAX_NUM_CANDS; i++)
  {
    RdModeList[i] = i;
  }

  //{
    static_vector<double, MRG_MAX_NUM_CANDS>  candCostList(MRG_MAX_NUM_CANDS, MAX_DOUBLE);
    // 1. Pass: get SATD-cost for selected candidates and reduce their count
    {
      const double sqrtLambdaForFirstPass = m_pcRdCost->getMotionLambda( );

      CodingUnit &cu = tempCS->addCU(CS::getArea(*tempCS, tempCS->area, (const ChannelType)partitioner.chType), (const ChannelType)partitioner.chType);

      partitioner.setCUData(cu);
      cu.slice = tempCS->slice;
      cu.tileIdx = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
      cu.skip = false;
      cu.predMode = MODE_IBC;
      cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
      cu.qp = encTestMode.qp;
      cu.mmvdSkip = false;
      cu.geoFlag = false;
      DistParam distParam;
      const bool bUseHadamard = !cu.slice->getDisableSATDForRD();
      PredictionUnit &pu = tempCS->addPU(cu, partitioner.chType); //tempCS->addPU(cu);
      pu.mmvdMergeFlag = false;
      pu.regularMergeFlag = false;
      Picture* refPic = pu.cu->slice->getPic();
      const CPelBuf refBuf = refPic->getRecoBuf(pu.blocks[COMPONENT_Y]);
      const Pel*        piRefSrch = refBuf.buf;
      if (tempCS->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())
      {
        const CompArea &area = cu.blocks[COMPONENT_Y];
        CompArea    tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
        PelBuf tmpLuma = m_tmpStorageLCU->getBuf(tmpArea);
        tmpLuma.copyFrom(tempCS->getOrgBuf().Y());
        tmpLuma.rspSignal(m_pcReshape->getFwdLUT());
        m_pcRdCost->setDistParam(distParam, tmpLuma, refBuf, sps.getBitDepth(CHANNEL_TYPE_LUMA), COMPONENT_Y, bUseHadamard);
      }
      else
      m_pcRdCost->setDistParam(distParam, tempCS->getOrgBuf().Y(), refBuf, sps.getBitDepth(CHANNEL_TYPE_LUMA), COMPONENT_Y, bUseHadamard);
      int refStride = refBuf.stride;
      const UnitArea localUnitArea(tempCS->area.chromaFormat, Area(0, 0, tempCS->area.Y().width, tempCS->area.Y().height));
      int numValidBv = mergeCtx.numValidMergeCand;
      for (unsigned int mergeCand = 0; mergeCand < mergeCtx.numValidMergeCand; mergeCand++)
      {
        mergeCtx.setMergeInfo(pu, mergeCand); // set bv info in merge mode
        const int cuPelX = pu.Y().x;
        const int cuPelY = pu.Y().y;
        int roiWidth = pu.lwidth();
        int roiHeight = pu.lheight();
        const int picWidth = pu.cs->slice->getPPS()->getPicWidthInLumaSamples();
        const int picHeight = pu.cs->slice->getPPS()->getPicHeightInLumaSamples();
        const unsigned int  lcuWidth = pu.cs->slice->getSPS()->getMaxCUWidth();
        int xPred = pu.bv.getHor();
        int yPred = pu.bv.getVer();

        if (!m_pcInterSearch->searchBv(pu, cuPelX, cuPelY, roiWidth, roiHeight, picWidth, picHeight, xPred, yPred, lcuWidth)) // not valid bv derived
        {
          numValidBv--;
          continue;
        }
        PU::spanMotionInfo(pu, mergeCtx);

        distParam.cur.buf = piRefSrch + refStride * yPred + xPred;

        Distortion sad = distParam.distFunc(distParam);
        unsigned int bitsCand = mergeCand + 1;
        if (mergeCand == tempCS->sps->getMaxNumMergeCand() - 1)
        {
          bitsCand--;
        }
        double cost = (double)sad + (double)bitsCand * sqrtLambdaForFirstPass;

        updateCandList(mergeCand, cost, RdModeList, candCostList
         , numMrgSATDCand);
      }

      // Try to limit number of candidates using SATD-costs
      if (numValidBv)
      {
        numMrgSATDCand = numValidBv;
        for (unsigned int i = 1; i < numValidBv; i++)
        {
          if (candCostList[i] > MRG_FAST_RATIO*candCostList[0])
          {
            numMrgSATDCand = i;
            break;
          }
        }
      }
      else
      {
        tempCS->dist = 0;
        tempCS->fracBits = 0;
        tempCS->cost = MAX_DOUBLE;
        tempCS->costDbOffset = 0;
        tempCS->initStructData(encTestMode.qp);
        return;
      }

      tempCS->initStructData(encTestMode.qp);
    }
  //}


  const unsigned int iteration = 2;
  m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;
  // 2. Pass: check candidates using full RD test
  for (unsigned int numResidualPass = 0; numResidualPass < iteration; numResidualPass++)
  {
    for (unsigned int mrgHADIdx = 0; mrgHADIdx < numMrgSATDCand; mrgHADIdx++)
    {
      unsigned int mergeCand = RdModeList[mrgHADIdx];
      if (!(numResidualPass == 1 && candHasNoResidual[mergeCand] == 1))
      {
        if (!(bestIsSkip && (numResidualPass == 0)))
        {
          {

            // first get merge candidates
            CodingUnit &cu = tempCS->addCU(CS::getArea(*tempCS, tempCS->area, (const ChannelType)partitioner.chType), (const ChannelType)partitioner.chType);

            partitioner.setCUData(cu);
            cu.slice = tempCS->slice;
            cu.tileIdx = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
            cu.skip = false;
            cu.predMode = MODE_IBC;
            cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
            cu.qp = encTestMode.qp;
            cu.sbtInfo = 0;

            PredictionUnit &pu = tempCS->addPU(cu, partitioner.chType);// tempCS->addPU(cu);
            pu.intraDir[0] = DC_IDX; // set intra pred for ibc block
            pu.intraDir[1] = PLANAR_IDX; // set intra pred for ibc block
            cu.mmvdSkip = false;
            pu.mmvdMergeFlag = false;
            pu.regularMergeFlag = false;
            cu.geoFlag = false;
            mergeCtx.setMergeInfo(pu, mergeCand);
            PU::spanMotionInfo(pu, mergeCtx);

            assert(mergeCtx.mrgTypeNeighbours[mergeCand] == MRG_TYPE_IBC); //  should be IBC candidate at this round
            const bool chroma = !pu.cu->isSepTree();

            //  MC
            m_pcInterSearch->motionCompensation(pu,REF_PIC_LIST_0, true, chroma);
            m_CABACEstimator->getCtx() = m_CurrCtx->start;

            m_pcInterSearch->encodeResAndCalcRdInterCU(
#if ANNM
              flag,
#endif 
              *tempCS, partitioner, (numResidualPass != 0), true, chroma);
            if (tempCS->slice->getSPS()->getUseColorTrans())
            {
              bestCS->tmpColorSpaceCost = tempCS->tmpColorSpaceCost;
              bestCS->firstColorSpaceSelected = tempCS->firstColorSpaceSelected;
            }
            xEncodeDontSplit(*tempCS, partitioner);

#if ENABLE_QPA_SUB_CTU
            xCheckDQP (*tempCS, partitioner);
#else
            // this if-check is redundant
            if (tempCS->pps->getUseDQP() && partitioner.currQgEnable())
            {
              xCheckDQP(*tempCS, partitioner);
            }
#endif
            xCheckChromaQPOffset( *tempCS, partitioner );


            DTRACE_MODE_COST(*tempCS, m_pcRdCost->getLambda());
            xCheckBestMode(tempCS, bestCS, partitioner, encTestMode);

            tempCS->initStructData(encTestMode.qp);
          }

            if (m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip)
            {
              if (bestCS->getCU(partitioner.chType) == NULL)
                bestIsSkip = 0;
              else
              bestIsSkip = bestCS->getCU(partitioner.chType)->rootCbf == 0;
            }
        }
      }
    }
  }
  if ( m_bestModeUpdated && bestCS->cost != MAX_DOUBLE )
  {
    xCalDebCost( *bestCS, partitioner );
  }
}

void EncCu::xCheckRDCostIBCMode(
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode)
{
  if (tempCS->area.lwidth() == 128 || tempCS->area.lheight() == 128) // disable IBC mode larger than 64x64
  {
    return;
  }

    tempCS->initStructData(encTestMode.qp);

    m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;

    CodingUnit &cu = tempCS->addCU(CS::getArea(*tempCS, tempCS->area, partitioner.chType), partitioner.chType);

    partitioner.setCUData(cu);
    cu.slice = tempCS->slice;
    cu.tileIdx = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
    cu.skip = false;
    cu.predMode = MODE_IBC;
    cu.chromaQpAdj = m_cuChromaQpOffsetIdxPlus1;
    cu.qp = encTestMode.qp;
    cu.imv = 0;
    cu.sbtInfo = 0;

    CU::addPUs(cu);

    m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;

    PredictionUnit& pu = *cu.firstPU;
    cu.mmvdSkip = false;
    pu.mmvdMergeFlag = false;
    pu.regularMergeFlag = false;

    pu.intraDir[0] = DC_IDX; // set intra pred for ibc block
    pu.intraDir[1] = PLANAR_IDX; // set intra pred for ibc block

    pu.interDir = 1; // use list 0 for IBC mode
    pu.refIdx[REF_PIC_LIST_0] = MAX_NUM_REF; // last idx in the list
      bool bValid = m_pcInterSearch->predIBCSearch(cu, partitioner, m_ctuIbcSearchRangeX, m_ctuIbcSearchRangeY, m_ibcHashMap);

      if (bValid)
      {
        PU::spanMotionInfo(pu);
        const bool chroma = !pu.cu->isSepTree();
        //  MC
        m_pcInterSearch->motionCompensation(pu, REF_PIC_LIST_0, true, chroma);

        {

          m_pcInterSearch->encodeResAndCalcRdInterCU(
#if ANNM
            flag,
#endif 
            *tempCS, partitioner, false, true, chroma);
          if (tempCS->slice->getSPS()->getUseColorTrans())
          {
            bestCS->tmpColorSpaceCost = tempCS->tmpColorSpaceCost;
            bestCS->firstColorSpaceSelected = tempCS->firstColorSpaceSelected;
          }

          xEncodeDontSplit(*tempCS, partitioner);

#if ENABLE_QPA_SUB_CTU
          xCheckDQP (*tempCS, partitioner);
#else
          // this if-check is redundant
          if (tempCS->pps->getUseDQP() && partitioner.currQgEnable())
          {
            xCheckDQP(*tempCS, partitioner);
          }
#endif
          xCheckChromaQPOffset( *tempCS, partitioner );

          tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();
          if ( m_bestModeUpdated )
          {
            xCalDebCost( *tempCS, partitioner );
          }

          DTRACE_MODE_COST(*tempCS, m_pcRdCost->getLambda());
          xCheckBestMode(tempCS, bestCS, partitioner, encTestMode);

        }

      } // bValid
      else
      {
        tempCS->dist = 0;
        tempCS->fracBits = 0;
        tempCS->cost = MAX_DOUBLE;
        tempCS->costDbOffset = 0;
      }
}
  // check ibc mode in encoder RD
  //////////////////////////////////////////////////////////////////////////////////////////////

void EncCu::xCheckRDCostInter(
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode)
{
  tempCS->initStructData( encTestMode.qp );


  m_pcInterSearch->setAffineModeSelected(false);

  if( tempCS->slice->getCheckLDC() )
  {
    m_bestBcwCost[0] = m_bestBcwCost[1] = std::numeric_limits<double>::max();
    m_bestBcwIdx[0] = m_bestBcwIdx[1] = -1;
  }

  m_pcInterSearch->resetBufferedUniMotions();
  int bcwLoopNum = (tempCS->slice->isInterB() ? BCW_NUM : 1);
  bcwLoopNum = (tempCS->sps->getUseBcw() ? bcwLoopNum : 1);

  if( tempCS->area.lwidth() * tempCS->area.lheight() < BCW_SIZE_CONSTRAINT )
  {
    bcwLoopNum = 1;
  }

  double curBestCost = bestCS->cost;
  double equBcwCost = MAX_DOUBLE;

  m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;

  for( int bcwLoopIdx = 0; bcwLoopIdx < bcwLoopNum; bcwLoopIdx++ )
  {
    if( m_pcEncCfg->getUseBcwFast() )
    {
      auto blkCache = dynamic_cast< CacheBlkInfoCtrl* >(m_modeCtrl);

      if( blkCache )
      {
        bool isBestInter = blkCache->getInter(bestCS->area);
        uint8_t bestBcwIdx = blkCache->getBcwIdx(bestCS->area);

        if( isBestInter && g_BcwSearchOrder[bcwLoopIdx] != BCW_DEFAULT && g_BcwSearchOrder[bcwLoopIdx] != bestBcwIdx )
        {
          continue;
        }
      }
    }
    if( !tempCS->slice->getCheckLDC() )
    {
      if( bcwLoopIdx != 0 && bcwLoopIdx != 3 && bcwLoopIdx != 4 )
      {
        continue;
      }
    }

  CodingUnit &cu      = tempCS->addCU( tempCS->area, partitioner.chType );

  partitioner.setCUData( cu );
  cu.slice            = tempCS->slice;
  cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
  cu.skip             = false;
  cu.mmvdSkip = false;
//cu.affine
  cu.predMode         = MODE_INTER;
  cu.chromaQpAdj      = m_cuChromaQpOffsetIdxPlus1;
  cu.qp               = encTestMode.qp;
  CU::addPUs( cu );

  cu.BcwIdx = g_BcwSearchOrder[bcwLoopIdx];
  uint8_t bcwIdx = cu.BcwIdx;
  bool  testBcw = (bcwIdx != BCW_DEFAULT);

  m_pcInterSearch->predInterSearch( cu, partitioner );

  bcwIdx = CU::getValidBcwIdx(cu);
  if( testBcw && bcwIdx == BCW_DEFAULT ) // Enabled Bcw but the search results is uni.
  {
    tempCS->initStructData(encTestMode.qp);
    continue;
  }
  CHECK_VTM(!(testBcw || (!testBcw && bcwIdx == BCW_DEFAULT)), " !( bTestBcw || (!bTestBcw && bcwIdx == BCW_DEFAULT ) )");

  bool isEqualUni = false;
  if( m_pcEncCfg->getUseBcwFast() )
  {
    if( cu.firstPU->interDir != 3 && testBcw == 0 )
    {
      isEqualUni = true;
    }
  }

  xEncodeInterResidual( 
#if ANNM
    flag,
#endif
    tempCS, bestCS, partitioner, encTestMode, 0
                        , 0
                        , &equBcwCost
  );

  if( g_BcwSearchOrder[bcwLoopIdx] == BCW_DEFAULT )
    m_pcInterSearch->setAffineModeSelected((bestCS->cus.front()->affine && !(bestCS->cus.front()->firstPU->mergeFlag)));

  tempCS->initStructData(encTestMode.qp);

  double skipTH = MAX_DOUBLE;
  skipTH = (m_pcEncCfg->getUseBcwFast() ? 1.05 : MAX_DOUBLE);
  if( equBcwCost > curBestCost * skipTH )
  {
    break;
  }

  if( m_pcEncCfg->getUseBcwFast() )
  {
    if( isEqualUni == true && m_pcEncCfg->getIntraPeriod() == -1 )
    {
      break;
    }
  }
  if( g_BcwSearchOrder[bcwLoopIdx] == BCW_DEFAULT && xIsBcwSkip(cu) && m_pcEncCfg->getUseBcwFast() )
  {
    break;
  }
 }  // for( UChar bcwLoopIdx = 0; bcwLoopIdx < bcwLoopNum; bcwLoopIdx++ )
  if ( m_bestModeUpdated && bestCS->cost != MAX_DOUBLE )
  {
    xCalDebCost( *bestCS, partitioner );
  }
}




bool EncCu::xCheckRDCostInterIMV(
#if ANNM
  bool& flag,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner, const EncTestMode &encTestMode,    \
  double &bestIntPelCost)
{
  int iIMV = int( ( encTestMode.opts & ETO_IMV ) >> ETO_IMV_SHIFT );
  m_pcInterSearch->setAffineModeSelected(false);
  // Only Half-Pel, int-Pel, 4-Pel and fast 4-Pel allowed
  CHECK_VTM(iIMV < 1 || iIMV > 4, "Unsupported IMV Mode");
  const bool testAltHpelFilter = iIMV == 4;
  // Fast 4-Pel Mode

  m_bestModeUpdated = tempCS->useDbCost = bestCS->useDbCost = false;

  EncTestMode encTestModeBase = encTestMode;                                        // copy for clearing non-IMV options
  encTestModeBase.opts        = EncTestModeOpts( encTestModeBase.opts & ETO_IMV );  // clear non-IMV options (is that intended?)

  tempCS->initStructData( encTestMode.qp );

  m_pcInterSearch->resetBufferedUniMotions();
  int bcwLoopNum = (tempCS->slice->isInterB() ? BCW_NUM : 1);
  bcwLoopNum = (tempCS->slice->getSPS()->getUseBcw() ? bcwLoopNum : 1);

  if( tempCS->area.lwidth() * tempCS->area.lheight() < BCW_SIZE_CONSTRAINT )
  {
    bcwLoopNum = 1;
  }

  bool validMode = false;
  double curBestCost = bestCS->cost;
  double equBcwCost = MAX_DOUBLE;

  for( int bcwLoopIdx = 0; bcwLoopIdx < bcwLoopNum; bcwLoopIdx++ )
  {
    if( m_pcEncCfg->getUseBcwFast() )
    {
      auto blkCache = dynamic_cast< CacheBlkInfoCtrl* >(m_modeCtrl);

      if( blkCache )
      {
        bool isBestInter = blkCache->getInter(bestCS->area);
        uint8_t bestBcwIdx = blkCache->getBcwIdx(bestCS->area);

        if( isBestInter && g_BcwSearchOrder[bcwLoopIdx] != BCW_DEFAULT && g_BcwSearchOrder[bcwLoopIdx] != bestBcwIdx )
        {
          continue;
        }
      }
    }

    if( !tempCS->slice->getCheckLDC() )
    {
      if( bcwLoopIdx != 0 && bcwLoopIdx != 3 && bcwLoopIdx != 4 )
      {
        continue;
      }
    }

    if( m_pcEncCfg->getUseBcwFast() && tempCS->slice->getCheckLDC() && g_BcwSearchOrder[bcwLoopIdx] != BCW_DEFAULT
      && (m_bestBcwIdx[0] >= 0 && g_BcwSearchOrder[bcwLoopIdx] != m_bestBcwIdx[0])
      && (m_bestBcwIdx[1] >= 0 && g_BcwSearchOrder[bcwLoopIdx] != m_bestBcwIdx[1]))
    {
      continue;
    }

  CodingUnit &cu = tempCS->addCU( tempCS->area, partitioner.chType );

  partitioner.setCUData( cu );
  cu.slice            = tempCS->slice;
  cu.tileIdx          = tempCS->pps->getTileIdx( tempCS->area.lumaPos() );
  cu.skip             = false;
  cu.mmvdSkip = false;
//cu.affine
  cu.predMode         = MODE_INTER;
  cu.chromaQpAdj      = m_cuChromaQpOffsetIdxPlus1;
  cu.qp               = encTestMode.qp;

  CU::addPUs( cu );

  if (testAltHpelFilter)
  {
    cu.imv = IMV_HPEL;
  }
  else
  {
    cu.imv = iIMV == 1 ? IMV_FPEL : IMV_4PEL;
  }

  bool testBcw;
  uint8_t bcwIdx;
  bool affineAmvrEanbledFlag = !testAltHpelFilter && cu.slice->getSPS()->getAffineAmvrEnabledFlag();

  cu.BcwIdx = g_BcwSearchOrder[bcwLoopIdx];
  bcwIdx = cu.BcwIdx;
  testBcw = (bcwIdx != BCW_DEFAULT);

  cu.firstPU->interDir = 10;

  m_pcInterSearch->predInterSearch( cu, partitioner );

  if ( cu.firstPU->interDir <= 3 )
  {
    bcwIdx = CU::getValidBcwIdx(cu);
  }
  else
  {
    return false;
  }

  if( m_pcEncCfg->getMCTSEncConstraint() && ( ( cu.firstPU->refIdx[L0] < 0 && cu.firstPU->refIdx[L1] < 0 ) || ( !( MCTSHelper::checkMvBufferForMCTSConstraint( *cu.firstPU ) ) ) ) )
  {
    // Do not use this mode
    tempCS->initStructData( encTestMode.qp );
    continue;
  }
  if( testBcw && bcwIdx == BCW_DEFAULT ) // Enabled Bcw but the search results is uni.
  {
    tempCS->initStructData(encTestMode.qp);
    continue;
  }
  CHECK_VTM(!(testBcw || (!testBcw && bcwIdx == BCW_DEFAULT)), " !( bTestBcw || (!bTestBcw && bcwIdx == BCW_DEFAULT ) )");

  bool isEqualUni = false;
  if( m_pcEncCfg->getUseBcwFast() )
  {
    if( cu.firstPU->interDir != 3 && testBcw == 0 )
    {
      isEqualUni = true;
    }
  }

  if ( !CU::hasSubCUNonZeroMVd( cu ) && !CU::hasSubCUNonZeroAffineMVd( cu ) )
  {
    if (m_modeCtrl->useModeResult(encTestModeBase, tempCS, partitioner))
    {
      std::swap(tempCS, bestCS);
      // store temp best CI for next CU coding
      m_CurrCtx->best = m_CABACEstimator->getCtx();
    }
    if ( affineAmvrEanbledFlag )
    {
      tempCS->initStructData( encTestMode.qp );
      continue;
    }
    else
    {
      return false;
    }
  }

  xEncodeInterResidual( 
#if ANNM
    flag,
#endif
    tempCS, bestCS, partitioner, encTestModeBase, 0
                        , 0
                        , &equBcwCost
  );

  if( cu.imv == IMV_FPEL && tempCS->cost < bestIntPelCost )
  {
    bestIntPelCost = tempCS->cost;
  }
  tempCS->initStructData(encTestMode.qp);

  double skipTH = MAX_DOUBLE;
  skipTH = (m_pcEncCfg->getUseBcwFast() ? 1.05 : MAX_DOUBLE);
  if( equBcwCost > curBestCost * skipTH )
  {
    break;
  }

  if( m_pcEncCfg->getUseBcwFast() )
  {
    if( isEqualUni == true && m_pcEncCfg->getIntraPeriod() == -1 )
    {
      break;
    }
  }
  if( g_BcwSearchOrder[bcwLoopIdx] == BCW_DEFAULT && xIsBcwSkip(cu) && m_pcEncCfg->getUseBcwFast() )
  {
    break;
  }
  validMode = true;
 } // for( UChar bcwLoopIdx = 0; bcwLoopIdx < bcwLoopNum; bcwLoopIdx++ )

  if ( m_bestModeUpdated && bestCS->cost != MAX_DOUBLE )
  {
    xCalDebCost( *bestCS, partitioner );
  }

  return tempCS->slice->getSPS()->getAffineAmvrEnabledFlag() ? validMode : true;
}

void EncCu::xCalDebCost( CodingStructure &cs, Partitioner &partitioner, bool calDist )
{
  if ( cs.cost == MAX_DOUBLE )
  {
    cs.costDbOffset = 0;
  }

  if ( cs.slice->getDeblockingFilterDisable() || ( !m_pcEncCfg->getUseEncDbOpt() && !calDist ) )
  {
    return;
  }

  m_pcLoopFilter->setEnc(true);
  const ChromaFormat format = cs.area.chromaFormat;
  CodingUnit*                cu = cs.getCU(partitioner.chType);
  const Position lumaPos = cu->Y().valid() ? cu->Y().pos() : recalcPosition( format, cu->chType, CHANNEL_TYPE_LUMA, cu->blocks[cu->chType].pos() );
  bool topEdgeAvai = lumaPos.y > 0 && ((lumaPos.y % 4) == 0);
  bool leftEdgeAvai = lumaPos.x > 0 && ((lumaPos.x % 4) == 0);
  bool anyEdgeAvai = topEdgeAvai || leftEdgeAvai;
  cs.costDbOffset = 0;

  if ( calDist )
  {
    const UnitArea currCsArea = clipArea( CS::getArea( cs, cs.area, partitioner.chType ), *cs.picture );
    ComponentID compStr = ( cu->isSepTree() && !isLuma( partitioner.chType ) ) ? COMPONENT_Cb : COMPONENT_Y;
    ComponentID compEnd = ( cu->isSepTree() && isLuma( partitioner.chType ) ) ? COMPONENT_Y : COMPONENT_Cr;
    Distortion finalDistortion = 0;
    for ( int comp = compStr; comp <= compEnd; comp++ )
    {
      const ComponentID compID = ComponentID( comp );
      CPelBuf org = cs.getOrgBuf( compID );
      CPelBuf reco = cs.getRecoBuf( compID );
      finalDistortion += getDistortionDb( cs, org, reco, compID, currCsArea.block( compID ), false );
    }
    //updated distortion
    cs.dist = finalDistortion;
  }

  if ( anyEdgeAvai && m_pcEncCfg->getUseEncDbOpt() )
  {
    ComponentID compStr = ( cu->isSepTree() && !isLuma( partitioner.chType ) ) ? COMPONENT_Cb : COMPONENT_Y;
    ComponentID compEnd = ( cu->isSepTree() &&  isLuma( partitioner.chType ) ) ? COMPONENT_Y : COMPONENT_Cr;

    const UnitArea currCsArea = clipArea( CS::getArea( cs, cs.area, partitioner.chType ), *cs.picture );

    PelStorage&          picDbBuf = m_pcLoopFilter->getDbEncPicYuvBuffer();

    //deblock neighbour pixels
    const Size     lumaSize = cu->Y().valid() ? cu->Y().size() : recalcSize( format, cu->chType, CHANNEL_TYPE_LUMA, cu->blocks[cu->chType].size() );

    const int verOffset = lumaPos.y > 7 ? 8 : 4;
    const int horOffset = lumaPos.x > 7 ? 8 : 4;
    const UnitArea areaTop(  format, Area( lumaPos.x, lumaPos.y - verOffset, lumaSize.width, verOffset  ) );
    const UnitArea areaLeft( format, Area( lumaPos.x - horOffset, lumaPos.y, horOffset, lumaSize.height ) );
    for ( int compIdx = compStr; compIdx <= compEnd; compIdx++ )
    {
      ComponentID compId = (ComponentID)compIdx;

      //Copy current CU's reco to Deblock Pic Buffer
      const CompArea&  curCompArea = currCsArea.block( compId );
      picDbBuf.getBuf( curCompArea ).copyFrom( cs.getRecoBuf( curCompArea ) );
      if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getSliceReshaperInfo().getUseSliceReshaper() && isLuma(compId))
      {
        picDbBuf.getBuf( curCompArea ).rspSignal( m_pcReshape->getInvLUT() );
      }

      //left neighbour
      if ( leftEdgeAvai )
      {
        const CompArea&  compArea = areaLeft.block(compId);
        picDbBuf.getBuf( compArea ).copyFrom( cs.picture->getRecoBuf( compArea ) );
        if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getSliceReshaperInfo().getUseSliceReshaper() && isLuma(compId))
        {
          picDbBuf.getBuf( compArea ).rspSignal( m_pcReshape->getInvLUT() );
        }
      }
      //top neighbour
      if ( topEdgeAvai )
      {
        const CompArea&  compArea = areaTop.block( compId );
        picDbBuf.getBuf( compArea ).copyFrom( cs.picture->getRecoBuf( compArea ) );
        if (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getSliceReshaperInfo().getUseSliceReshaper() && isLuma(compId))
        {
          picDbBuf.getBuf( compArea ).rspSignal( m_pcReshape->getInvLUT() );
        }
      }
    }

    //deblock
    if ( leftEdgeAvai )
    {
      m_pcLoopFilter->resetFilterLengths();
      m_pcLoopFilter->xDeblockCU( *cu, EDGE_VER );
    }

    if (topEdgeAvai)
    {
      m_pcLoopFilter->resetFilterLengths();
      m_pcLoopFilter->xDeblockCU( *cu, EDGE_HOR );
    }

    //update current CU SSE
    Distortion distCur = 0;
    for ( int compIdx = compStr; compIdx <= compEnd; compIdx++ )
    {
      ComponentID compId = (ComponentID)compIdx;
      CPelBuf reco = picDbBuf.getBuf( currCsArea.block( compId ) );
      CPelBuf org = cs.getOrgBuf( compId );
      distCur += getDistortionDb( cs, org, reco, compId, currCsArea.block( compId ), true );
    }

    //calculate difference between DB_before_SSE and DB_after_SSE for neighbouring CUs
    Distortion distBeforeDb = 0, distAfterDb = 0;
    for (int compIdx = compStr; compIdx <= compEnd; compIdx++)
    {
      ComponentID compId = (ComponentID)compIdx;
      if ( leftEdgeAvai )
      {
        const CompArea&  compArea = areaLeft.block( compId );
        CPelBuf org = cs.picture->getOrigBuf( compArea );
        CPelBuf reco = cs.picture->getRecoBuf( compArea );
        CPelBuf recoDb = picDbBuf.getBuf( compArea );
        distBeforeDb += getDistortionDb( cs, org, reco,   compId, compArea, false );
        distAfterDb  += getDistortionDb( cs, org, recoDb, compId, compArea, true  );
      }
      if ( topEdgeAvai )
      {
        const CompArea&  compArea = areaTop.block( compId );
        CPelBuf org = cs.picture->getOrigBuf( compArea );
        CPelBuf reco = cs.picture->getRecoBuf( compArea );
        CPelBuf recoDb = picDbBuf.getBuf( compArea );
        distBeforeDb += getDistortionDb( cs, org, reco,   compId, compArea, false );
        distAfterDb  += getDistortionDb( cs, org, recoDb, compId, compArea, true  );
      }
    }

    //updated cost
    int64_t distTmp = distCur - cs.dist + distAfterDb - distBeforeDb;
    int sign = distTmp < 0 ? -1 : 1;
    distTmp = distTmp < 0 ? -distTmp : distTmp;
    cs.costDbOffset = sign * m_pcRdCost->calcRdCost( 0, distTmp );
  }

  m_pcLoopFilter->setEnc( false );
}

Distortion EncCu::getDistortionDb( CodingStructure &cs, CPelBuf org, CPelBuf reco, ComponentID compID, const CompArea& compArea, bool afterDb )
{
  Distortion dist = 0;
#if WCG_EXT
  m_pcRdCost->setChromaFormat(cs.sps->getChromaFormatIdc());
  CPelBuf orgLuma = cs.picture->getOrigBuf( cs.area.blocks[COMPONENT_Y] );
  if (m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() || (
    m_pcEncCfg->getLmcs() && (cs.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())))
  {
    if ( compID == COMPONENT_Y && !afterDb && !m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled())
    {
      CompArea    tmpArea( COMPONENT_Y, cs.area.chromaFormat, Position( 0, 0 ), compArea.size() );
      PelBuf tmpRecLuma = m_tmpStorageLCU->getBuf( tmpArea );
      tmpRecLuma.copyFrom( reco );
      tmpRecLuma.rspSignal( m_pcReshape->getInvLUT() );
      dist += m_pcRdCost->getDistPart( org, tmpRecLuma, cs.sps->getBitDepth( toChannelType( compID ) ), compID, DF_SSE_WTD, &orgLuma );
    }
    else
    {
      dist += m_pcRdCost->getDistPart( org, reco, cs.sps->getBitDepth( toChannelType( compID ) ), compID, DF_SSE_WTD, &orgLuma );
    }
  }
  else if (m_pcEncCfg->getLmcs() && cs.slice->getLmcsEnabledFlag() && cs.slice->isIntra()) //intra slice
  {
    if ( compID == COMPONENT_Y && afterDb )
    {
      CompArea    tmpArea( COMPONENT_Y, cs.area.chromaFormat, Position( 0, 0 ), compArea.size() );
      PelBuf tmpRecLuma = m_tmpStorageLCU->getBuf( tmpArea );
      tmpRecLuma.copyFrom( reco );
      tmpRecLuma.rspSignal( m_pcReshape->getFwdLUT() );
      dist += m_pcRdCost->getDistPart( org, tmpRecLuma, cs.sps->getBitDepth( toChannelType( compID ) ), compID, DF_SSE );
    }
    else
    {
      dist += m_pcRdCost->getDistPart( org, reco, cs.sps->getBitDepth(toChannelType( compID ) ), compID, DF_SSE );
    }
  }
  else
#endif
  {
    dist = m_pcRdCost->getDistPart( org, reco, cs.sps->getBitDepth( toChannelType( compID ) ), compID, DF_SSE );
  }
  return dist;
}

void EncCu::xEncodeInterResidual(  
#if ANNM
  bool &flag,
#endif
  CodingStructure *&tempCS
                                  , CodingStructure *&bestCS
                                  , Partitioner &partitioner
                                  , const EncTestMode& encTestMode
                                  , int residualPass
                                  , bool* bestHasNonResi
                                  , double* equBcwCost
  )
{

  CodingUnit*            cu        = tempCS->getCU( partitioner.chType );
  double   bestCostInternal        = MAX_DOUBLE;
  double           bestCost        = bestCS->cost;
  double           bestCostBegin   = bestCS->cost;
  CodingUnit*      prevBestCU      = bestCS->getCU( partitioner.chType );
  uint8_t          prevBestSbt     = ( prevBestCU == nullptr ) ? 0 : prevBestCU->sbtInfo;
  bool              swapped        = false; // avoid unwanted data copy
  bool             reloadCU        = false;

  const PredictionUnit& pu = *cu->firstPU;

  // clang-format off
  const int affineShiftTab[3] =
  {
    MV_PRECISION_INTERNAL - MV_PRECISION_QUARTER,
    MV_PRECISION_INTERNAL - MV_PRECISION_SIXTEENTH,
    MV_PRECISION_INTERNAL - MV_PRECISION_INT
  };

  const int normalShiftTab[NUM_IMV_MODES] =
  {
    MV_PRECISION_INTERNAL - MV_PRECISION_QUARTER,
    MV_PRECISION_INTERNAL - MV_PRECISION_INT,
    MV_PRECISION_INTERNAL - MV_PRECISION_4PEL,
    MV_PRECISION_INTERNAL - MV_PRECISION_HALF,
  };
  // clang-format on

  int mvShift;

  for (int refList = 0; refList < NUM_REF_PIC_LIST_01; refList++)
  {
    if (pu.refIdx[refList] >= 0)
    {
      if (!cu->affine)
      {
        mvShift = normalShiftTab[cu->imv];
        Mv signaledmvd(pu.mvd[refList].getHor() >> mvShift, pu.mvd[refList].getVer() >> mvShift);
        if (!((signaledmvd.getHor() >= MVD_MIN) && (signaledmvd.getHor() <= MVD_MAX)) || !((signaledmvd.getVer() >= MVD_MIN) && (signaledmvd.getVer() <= MVD_MAX)))
          return;
      }
      else
      {
        for (int ctrlP = 1 + (cu->affineType == AFFINEMODEL_6PARAM); ctrlP >= 0; ctrlP--)
        {
          mvShift = affineShiftTab[cu->imv];
          Mv signaledmvd(pu.mvdAffi[refList][ctrlP].getHor() >> mvShift, pu.mvdAffi[refList][ctrlP].getVer() >> mvShift);
          if (!((signaledmvd.getHor() >= MVD_MIN) && (signaledmvd.getHor() <= MVD_MAX)) || !((signaledmvd.getVer() >= MVD_MIN) && (signaledmvd.getVer() <= MVD_MAX)))
            return;
        }
      }
    }
  }
  // avoid MV exceeding 18-bit dynamic range
  const int maxMv = 1 << 17;
  if (!cu->affine && !pu.mergeFlag)
  {
    if ( (pu.refIdx[0] >= 0 && (pu.mv[0].getAbsHor() >= maxMv || pu.mv[0].getAbsVer() >= maxMv))
      || (pu.refIdx[1] >= 0 && (pu.mv[1].getAbsHor() >= maxMv || pu.mv[1].getAbsVer() >= maxMv)))
    {
      return;
    }
  }
  if (cu->affine && !pu.mergeFlag)
  {
    for (int refList = 0; refList < NUM_REF_PIC_LIST_01; refList++)
    {
      if (pu.refIdx[refList] >= 0)
      {
        for (int ctrlP = 1 + (cu->affineType == AFFINEMODEL_6PARAM); ctrlP >= 0; ctrlP--)
        {
          if (pu.mvAffi[refList][ctrlP].getAbsHor() >= maxMv || pu.mvAffi[refList][ctrlP].getAbsVer() >= maxMv)
          {
            return;
          }
        }
      }
    }
  }
  const bool mtsAllowed = tempCS->sps->getUseInterMTS() && CU::isInter( *cu ) && partitioner.currArea().lwidth() <= MTS_INTER_MAX_CU_SIZE && partitioner.currArea().lheight() <= MTS_INTER_MAX_CU_SIZE;
  uint8_t sbtAllowed = cu->checkAllowedSbt();
  //SBT resolution-dependent fast algorithm: not try size-64 SBT in RDO for low-resolution sequences (now resolution below HD)
  if( tempCS->pps->getPicWidthInLumaSamples() < (uint32_t)m_pcEncCfg->getSBTFast64WidthTh() )
  {
    sbtAllowed = ((cu->lwidth() > 32 || cu->lheight() > 32)) ? 0 : sbtAllowed;
  }
  uint8_t numRDOTried = 0;
  Distortion sbtOffDist = 0;
  bool    sbtOffRootCbf = 0;
  double  sbtOffCost      = MAX_DOUBLE;
  double  currBestCost = MAX_DOUBLE;
  bool    doPreAnalyzeResi = ( sbtAllowed || mtsAllowed ) && residualPass == 0;

  m_pcInterSearch->initTuAnalyzer();
  if( doPreAnalyzeResi )
  {
    m_pcInterSearch->calcMinDistSbt( *tempCS, *cu, sbtAllowed );
  }

  auto    slsSbt = dynamic_cast<SaveLoadEncInfoSbt*>( m_modeCtrl );
  int     slShift = 4 + std::min( (int)gp_sizeIdxInfo->idxFrom( cu->lwidth() ) + (int)gp_sizeIdxInfo->idxFrom( cu->lheight() ), 9 );
  Distortion curPuSse = m_pcInterSearch->getEstDistSbt( NUMBER_SBT_MODE );
  uint8_t currBestSbt = 0;
  uint8_t currBestTrs = MAX_UCHAR;
  uint8_t histBestSbt = MAX_UCHAR;
  uint8_t histBestTrs = MAX_UCHAR;
  m_pcInterSearch->setHistBestTrs( MAX_UCHAR, MAX_UCHAR );
  if( doPreAnalyzeResi )
  {
    if( m_pcInterSearch->getSkipSbtAll() && !mtsAllowed ) //emt is off
    {
      histBestSbt = 0; //try DCT2
      m_pcInterSearch->setHistBestTrs( histBestSbt, histBestTrs );
    }
    else
    {
      assert( curPuSse != std::numeric_limits<uint64_t>::max() );
      uint16_t compositeSbtTrs = slsSbt->findBestSbt( cu->cs->area, (uint32_t)( curPuSse >> slShift ) );
      histBestSbt = ( compositeSbtTrs >> 0 ) & 0xff;
      histBestTrs = ( compositeSbtTrs >> 8 ) & 0xff;
      if( m_pcInterSearch->getSkipSbtAll() && CU::isSbtMode( histBestSbt ) ) //special case, skip SBT when loading SBT
      {
        histBestSbt = 0; //try DCT2
      }
      m_pcInterSearch->setHistBestTrs( histBestSbt, histBestTrs );
    }
  }

  {
    if( reloadCU )
    {
      if( bestCost == bestCS->cost ) //The first EMT pass didn't become the bestCS, so we clear the TUs generated
      {
        tempCS->clearTUs();
      }
      else if( false == swapped )
      {
        tempCS->initStructData( encTestMode.qp );
        tempCS->copyStructure( *bestCS, partitioner.chType );
        tempCS->getPredBuf().copyFrom( bestCS->getPredBuf() );
        bestCost = bestCS->cost;
        cu       = tempCS->getCU( partitioner.chType );
        swapped = true;
      }
      else
      {
        tempCS->clearTUs();
        bestCost = bestCS->cost;
        cu       = tempCS->getCU( partitioner.chType );
      }

      //we need to restart the distortion for the new tempCS, the bit count and the cost
      tempCS->dist     = 0;
      tempCS->fracBits = 0;
      tempCS->cost     = MAX_DOUBLE;
      tempCS->costDbOffset = 0;
    }

    reloadCU    = true; // enable cu reloading
    cu->skip    = false;
    cu->sbtInfo = 0;

    const bool skipResidual = residualPass == 1;
    if( skipResidual || histBestSbt == MAX_UCHAR || !CU::isSbtMode( histBestSbt ) )
    {
    m_pcInterSearch->encodeResAndCalcRdInterCU( 
#if ANNM
        flag,
#endif 
      *tempCS, partitioner, skipResidual);
    if (tempCS->slice->getSPS()->getUseColorTrans())
    {
      bestCS->tmpColorSpaceCost = tempCS->tmpColorSpaceCost;
      bestCS->firstColorSpaceSelected = tempCS->firstColorSpaceSelected;
    }
    numRDOTried += mtsAllowed ? 2 : 1;
    xEncodeDontSplit( *tempCS, partitioner );

    xCheckDQP( *tempCS, partitioner );
    xCheckChromaQPOffset( *tempCS, partitioner );


    if( NULL != bestHasNonResi && (bestCostInternal > tempCS->cost) )
    {
      bestCostInternal = tempCS->cost;
      if (!(tempCS->getPU(partitioner.chType)->ciipFlag))
      *bestHasNonResi  = !cu->rootCbf;
    }

    if (cu->rootCbf == false)
    {
      if (tempCS->getPU(partitioner.chType)->ciipFlag)
      {
        tempCS->cost = MAX_DOUBLE;
        tempCS->costDbOffset = 0;
        return;
      }
    }
    currBestCost = tempCS->cost;
    sbtOffCost = tempCS->cost;
    sbtOffDist = tempCS->dist;
    sbtOffRootCbf = cu->rootCbf;
    currBestSbt = CU::getSbtInfo(cu->firstTU->mtsIdx[COMPONENT_Y] > MTS_SKIP ? SBT_OFF_MTS : SBT_OFF_DCT, 0);
    currBestTrs = cu->firstTU->mtsIdx[COMPONENT_Y];

#if WCG_EXT
    DTRACE_MODE_COST( *tempCS, m_pcRdCost->getLambda( true ) );
#else
    DTRACE_MODE_COST( *tempCS, m_pcRdCost->getLambda() );
#endif
    xCheckBestMode( tempCS, bestCS, partitioner, encTestMode );

    }

    uint8_t numSbtRdo = CU::numSbtModeRdo( sbtAllowed );
    //early termination if all SBT modes are not allowed
    //normative
    if( !sbtAllowed || skipResidual )
    {
      numSbtRdo = 0;
    }
    //fast algorithm
    if( ( histBestSbt != MAX_UCHAR && !CU::isSbtMode( histBestSbt ) ) || m_pcInterSearch->getSkipSbtAll() )
    {
      numSbtRdo = 0;
    }
    if( bestCost != MAX_DOUBLE && sbtOffCost != MAX_DOUBLE )
    {
      double th = 1.07;
      if( !( prevBestSbt == 0 || m_sbtCostSave[0] == MAX_DOUBLE ) )
      {
        assert( m_sbtCostSave[1] <= m_sbtCostSave[0] );
        th *= ( m_sbtCostSave[0] / m_sbtCostSave[1] );
      }
      if( sbtOffCost > bestCost * th )
      {
        numSbtRdo = 0;
      }
    }
    if( !sbtOffRootCbf && sbtOffCost != MAX_DOUBLE )
    {
      double th = Clip3( 0.05, 0.55, ( 27 - cu->qp ) * 0.02 + 0.35 );
      if( sbtOffCost < m_pcRdCost->calcRdCost( ( cu->lwidth() * cu->lheight() ) << SCALE_BITS, 0 ) * th )
      {
        numSbtRdo = 0;
      }
    }

    if( histBestSbt != MAX_UCHAR && numSbtRdo != 0 )
    {
      numSbtRdo = 1;
      m_pcInterSearch->initSbtRdoOrder( CU::getSbtMode( CU::getSbtIdx( histBestSbt ), CU::getSbtPos( histBestSbt ) ) );
    }

    for( int sbtModeIdx = 0; sbtModeIdx < numSbtRdo; sbtModeIdx++ )
    {
      uint8_t sbtMode = m_pcInterSearch->getSbtRdoOrder( sbtModeIdx );
      uint8_t sbtIdx = CU::getSbtIdxFromSbtMode( sbtMode );
      uint8_t sbtPos = CU::getSbtPosFromSbtMode( sbtMode );

      //fast algorithm (early skip, save & load)
      if( histBestSbt == MAX_UCHAR )
      {
        uint8_t skipCode = m_pcInterSearch->skipSbtByRDCost( cu->lwidth(), cu->lheight(), cu->mtDepth, sbtIdx, sbtPos, bestCS->cost, sbtOffDist, sbtOffCost, sbtOffRootCbf );
        if( skipCode != MAX_UCHAR )
        {
          continue;
        }

        if( sbtModeIdx > 0 )
        {
          uint8_t prevSbtMode = m_pcInterSearch->getSbtRdoOrder( sbtModeIdx - 1 );
          //make sure the prevSbtMode is the same size as the current SBT mode (otherwise the estimated dist may not be comparable)
          if( CU::isSameSbtSize( prevSbtMode, sbtMode ) )
          {
            Distortion currEstDist = m_pcInterSearch->getEstDistSbt( sbtMode );
            Distortion prevEstDist = m_pcInterSearch->getEstDistSbt( prevSbtMode );
            if( currEstDist > prevEstDist * 1.15 )
            {
              continue;
            }
          }
        }
      }

      //init tempCS and TU
      if( bestCost == bestCS->cost ) //The first EMT pass didn't become the bestCS, so we clear the TUs generated
      {
        tempCS->clearTUs();
      }
      else if( false == swapped )
      {
        tempCS->initStructData( encTestMode.qp );
        tempCS->copyStructure( *bestCS, partitioner.chType );
        tempCS->getPredBuf().copyFrom( bestCS->getPredBuf() );
        bestCost = bestCS->cost;
        cu = tempCS->getCU( partitioner.chType );
        swapped = true;
      }
      else
      {
        tempCS->clearTUs();
        bestCost = bestCS->cost;
        cu = tempCS->getCU( partitioner.chType );
      }

      //we need to restart the distortion for the new tempCS, the bit count and the cost
      tempCS->dist = 0;
      tempCS->fracBits = 0;
      tempCS->cost = MAX_DOUBLE;
      cu->skip = false;

      //set SBT info
      cu->setSbtIdx( sbtIdx );
      cu->setSbtPos( sbtPos );

      //try residual coding
      m_pcInterSearch->encodeResAndCalcRdInterCU( 
#if ANNM
        flag,
#endif 
        *tempCS, partitioner, skipResidual);
      if (tempCS->slice->getSPS()->getUseColorTrans())
      {
        bestCS->tmpColorSpaceCost = tempCS->tmpColorSpaceCost;
        bestCS->firstColorSpaceSelected = tempCS->firstColorSpaceSelected;
      }
      numRDOTried++;

      xEncodeDontSplit( *tempCS, partitioner );

      xCheckDQP( *tempCS, partitioner );
      xCheckChromaQPOffset( *tempCS, partitioner );

      if( NULL != bestHasNonResi && ( bestCostInternal > tempCS->cost ) )
      {
        bestCostInternal = tempCS->cost;
        if( !( tempCS->getPU( partitioner.chType )->ciipFlag ) )
          *bestHasNonResi = !cu->rootCbf;
      }

      if( tempCS->cost < currBestCost )
      {
        currBestSbt = cu->sbtInfo;
        currBestTrs = tempCS->tus[cu->sbtInfo ? cu->getSbtPos() : 0]->mtsIdx[COMPONENT_Y];
        assert( currBestTrs == 0 || currBestTrs == 1 );
        currBestCost = tempCS->cost;
      }

#if WCG_EXT
      DTRACE_MODE_COST( *tempCS, m_pcRdCost->getLambda( true ) );
#else
      DTRACE_MODE_COST( *tempCS, m_pcRdCost->getLambda() );
#endif
      xCheckBestMode( tempCS, bestCS, partitioner, encTestMode );
    }

    if( bestCostBegin != bestCS->cost )
    {
      m_sbtCostSave[0] = sbtOffCost;
      m_sbtCostSave[1] = currBestCost;
    }
  } //end emt loop

  if( histBestSbt == MAX_UCHAR && doPreAnalyzeResi && numRDOTried > 1 )
  {
    slsSbt->saveBestSbt( cu->cs->area, (uint32_t)( curPuSse >> slShift ), currBestSbt, currBestTrs );
  }
  tempCS->cost = currBestCost;
  if( ETM_INTER_ME == encTestMode.type )
  {
    if( equBcwCost != NULL )
    {
      if( tempCS->cost < ( *equBcwCost ) && cu->BcwIdx == BCW_DEFAULT )
      {
        ( *equBcwCost ) = tempCS->cost;
      }
    }
    else
    {
      CHECK_VTM( equBcwCost == NULL, "equBcwCost == NULL" );
    }
    if( tempCS->slice->getCheckLDC() && !cu->imv && cu->BcwIdx != BCW_DEFAULT && tempCS->cost < m_bestBcwCost[1] )
    {
      if( tempCS->cost < m_bestBcwCost[0] )
      {
        m_bestBcwCost[1] = m_bestBcwCost[0];
        m_bestBcwCost[0] = tempCS->cost;
        m_bestBcwIdx[1] = m_bestBcwIdx[0];
        m_bestBcwIdx[0] = cu->BcwIdx;
      }
      else
      {
        m_bestBcwCost[1] = tempCS->cost;
        m_bestBcwIdx[1] = cu->BcwIdx;
      }
    }
  }
}


void EncCu::xEncodeDontSplit( CodingStructure &cs, Partitioner &partitioner )
{
  m_CABACEstimator->resetBits();

  m_CABACEstimator->split_cu_mode( CU_DONT_SPLIT, cs, partitioner );
  if( partitioner.treeType == TREE_C )
    CHECK_VTM( m_CABACEstimator->getEstFracBits() != 0, "must be 0 bit" );

  cs.fracBits += m_CABACEstimator->getEstFracBits(); // split bits
  cs.cost      = m_pcRdCost->calcRdCost( cs.fracBits, cs.dist );

}

#if REUSE_CU_RESULTS
void EncCu::xReuseCachedResult( 
#if ANNM
  Pel* img[], int pieX, int pieY,
#endif
  CodingStructure *&tempCS, CodingStructure *&bestCS, Partitioner &partitioner)
{
  m_pcRdCost->setChromaFormat(tempCS->sps->getChromaFormatIdc());
  BestEncInfoCache* bestEncCache = dynamic_cast<BestEncInfoCache*>( m_modeCtrl );
  CHECK_VTM( !bestEncCache, "If this mode is chosen, mode controller has to implement the mode caching capabilities" );
  EncTestMode cachedMode;

  if( bestEncCache->setCsFrom( *tempCS, cachedMode, partitioner ) )
  {
    CodingUnit& cu = *tempCS->cus.front();
    partitioner.setCUData( cu );

    if( CU::isIntra( cu )
    || CU::isPLT(cu)
    )
    {
      xReconIntraQT(
#if ANNM
        img, pieX, pieY,
#endif
        cu);
    }
    else
    {
      xDeriveCUMV( cu );
      xReconInter( 
#if ANNM
        img, pieX, pieY,
#endif
        cu);
    }

    Distortion finalDistortion = 0;
    tempCS->useDbCost = m_pcEncCfg->getUseEncDbOpt();
    if ( m_pcEncCfg->getUseEncDbOpt() )
    {
      xCalDebCost( *tempCS, partitioner, true );
      finalDistortion = tempCS->dist;
    }
    else
    {
    const SPS &sps = *tempCS->sps;
    const int  numValidComponents = getNumberValidComponents( tempCS->area.chromaFormat );

    for( int comp = 0; comp < numValidComponents; comp++ )
    {
      const ComponentID compID = ComponentID( comp );

      if( partitioner.isSepTree( *tempCS ) && toChannelType( compID ) != partitioner.chType )
      {
        continue;
      }

      CPelBuf reco = tempCS->getRecoBuf( compID );
      CPelBuf org  = tempCS->getOrgBuf ( compID );

#if WCG_EXT
      if (m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() || (
        m_pcEncCfg->getLmcs() && (tempCS->slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag())))
      {
        const CPelBuf orgLuma = tempCS->getOrgBuf(tempCS->area.blocks[COMPONENT_Y]);
        if (compID == COMPONENT_Y && !(m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled()))
        {
          const CompArea &area = cu.blocks[COMPONENT_Y];
          CompArea    tmpArea(COMPONENT_Y, area.chromaFormat, Position(0, 0), area.size());
          PelBuf tmpRecLuma = m_tmpStorageLCU->getBuf(tmpArea);
          tmpRecLuma.copyFrom(reco);
          tmpRecLuma.rspSignal(m_pcReshape->getInvLUT());
          finalDistortion += m_pcRdCost->getDistPart(org, tmpRecLuma, sps.getBitDepth(toChannelType(compID)), compID, DF_SSE_WTD, &orgLuma);
        }
        else
        finalDistortion += m_pcRdCost->getDistPart( org, reco, sps.getBitDepth( toChannelType( compID ) ), compID, DF_SSE_WTD, &orgLuma );
      }
      else
#endif
      finalDistortion += m_pcRdCost->getDistPart( org, reco, sps.getBitDepth( toChannelType( compID ) ), compID, DF_SSE );
    }
    }

    m_CABACEstimator->getCtx() = m_CurrCtx->start;
    m_CABACEstimator->resetBits();

    CUCtx cuCtx;
    cuCtx.isDQPCoded = true;
    cuCtx.isChromaQpAdjCoded = true;
#if ANNM
    bool flag0 = false;
#endif
    m_CABACEstimator->coding_unit( 
#if ANNM
      flag0,
#endif
      cu, partitioner, cuCtx);


    tempCS->dist     = finalDistortion;
    tempCS->fracBits = m_CABACEstimator->getEstFracBits();
    tempCS->cost     = m_pcRdCost->calcRdCost( tempCS->fracBits, tempCS->dist );

    xEncodeDontSplit( *tempCS,         partitioner );
    xCheckDQP       ( *tempCS,         partitioner );
    xCheckChromaQPOffset( *tempCS,     partitioner );
    xCheckBestMode  (  tempCS, bestCS, partitioner, cachedMode );
  }
  else
  {
    THROW( "Should never happen!" );
  }
}
#endif


//! \}
