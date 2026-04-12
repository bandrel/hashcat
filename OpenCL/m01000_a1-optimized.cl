/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

#define NEW_SIMD_CODE

#ifdef KERNEL_STATIC
#include M2S(INCLUDE_PATH/inc_vendor.h)
#include M2S(INCLUDE_PATH/inc_types.h)
#include M2S(INCLUDE_PATH/inc_platform.cl)
#include M2S(INCLUDE_PATH/inc_common.cl)
#include M2S(INCLUDE_PATH/inc_simd.cl)
#include M2S(INCLUDE_PATH/inc_hash_md4.cl)
#endif

KERNEL_FQ KERNEL_FA void m01000_m04 (KERN_ATTR_BASIC ())
{
  /**
   * modifier
   */

  const u64 lid = get_local_id (0);

  /**
   * base
   */

  const u64 gid = get_global_id (0);

  if (gid >= GID_CNT) return;

  u32 pw_buf0[4];
  u32 pw_buf1[4];

  pw_buf0[0] = pws[gid].i[0];
  pw_buf0[1] = pws[gid].i[1];
  pw_buf0[2] = pws[gid].i[2];
  pw_buf0[3] = pws[gid].i[3];
  pw_buf1[0] = pws[gid].i[4];
  pw_buf1[1] = pws[gid].i[5];
  pw_buf1[2] = pws[gid].i[6];
  pw_buf1[3] = pws[gid].i[7];

  const u32 pw_l_len = pws[gid].pw_len & 63;

  /**
   * loop
   */

  for (u32 il_pos = 0; il_pos < IL_CNT; il_pos += VECT_SIZE)
  {
    const u32x pw_r_len = pwlenx_create_combt (combs_buf, il_pos) & 63;

    const u32x pw_len = (pw_l_len + pw_r_len) & 63;

    /**
     * concat password candidate
     */

    u32x wordl0[4] = { 0 };
    u32x wordl1[4] = { 0 };
    u32x wordl2[4] = { 0 };
    u32x wordl3[4] = { 0 };

    wordl0[0] = pw_buf0[0];
    wordl0[1] = pw_buf0[1];
    wordl0[2] = pw_buf0[2];
    wordl0[3] = pw_buf0[3];
    wordl1[0] = pw_buf1[0];
    wordl1[1] = pw_buf1[1];
    wordl1[2] = pw_buf1[2];
    wordl1[3] = pw_buf1[3];

    u32x wordr0[4] = { 0 };
    u32x wordr1[4] = { 0 };
    u32x wordr2[4] = { 0 };
    u32x wordr3[4] = { 0 };

    wordr0[0] = ix_create_combt (combs_buf, il_pos, 0);
    wordr0[1] = ix_create_combt (combs_buf, il_pos, 1);
    wordr0[2] = ix_create_combt (combs_buf, il_pos, 2);
    wordr0[3] = ix_create_combt (combs_buf, il_pos, 3);
    wordr1[0] = ix_create_combt (combs_buf, il_pos, 4);
    wordr1[1] = ix_create_combt (combs_buf, il_pos, 5);
    wordr1[2] = ix_create_combt (combs_buf, il_pos, 6);
    wordr1[3] = ix_create_combt (combs_buf, il_pos, 7);

    if (COMBS_MODE == COMBINATOR_MODE_BASE_LEFT)
    {
      switch_buffer_by_offset_le_VV (wordr0, wordr1, wordr2, wordr3, pw_l_len);
    }
    else
    {
      switch_buffer_by_offset_le_VV (wordl0, wordl1, wordl2, wordl3, pw_r_len);
    }

    u32x w0[4];
    u32x w1[4];
    u32x w2[4];
    u32x w3[4];

    w0[0] = wordl0[0] | wordr0[0];
    w0[1] = wordl0[1] | wordr0[1];
    w0[2] = wordl0[2] | wordr0[2];
    w0[3] = wordl0[3] | wordr0[3];
    w1[0] = wordl1[0] | wordr1[0];
    w1[1] = wordl1[1] | wordr1[1];
    w1[2] = wordl1[2] | wordr1[2];
    w1[3] = wordl1[3] | wordr1[3];
    w2[0] = wordl2[0] | wordr2[0];
    w2[1] = wordl2[1] | wordr2[1];
    w2[2] = wordl2[2] | wordr2[2];
    w2[3] = wordl2[3] | wordr2[3];
    w3[0] = wordl3[0] | wordr3[0];
    w3[1] = wordl3[1] | wordr3[1];
    w3[2] = wordl3[2] | wordr3[2];
    w3[3] = wordl3[3] | wordr3[3];

    make_utf16le (w1, w2, w3);
    make_utf16le (w0, w0, w1);

    w3[2] = pw_len * 8 * 2;
    w3[3] = 0;

    /**
     * precompute w[i] + constant for all 48 MD4 steps
     */

    const u32x F_w0c00 = w0[0] + make_u32x (MD4C00);
    const u32x F_w1c00 = w0[1] + make_u32x (MD4C00);
    const u32x F_w2c00 = w0[2] + make_u32x (MD4C00);
    const u32x F_w3c00 = w0[3] + make_u32x (MD4C00);
    const u32x F_w4c00 = w1[0] + make_u32x (MD4C00);
    const u32x F_w5c00 = w1[1] + make_u32x (MD4C00);
    const u32x F_w6c00 = w1[2] + make_u32x (MD4C00);
    const u32x F_w7c00 = w1[3] + make_u32x (MD4C00);
    const u32x F_w8c00 = w2[0] + make_u32x (MD4C00);
    const u32x F_w9c00 = w2[1] + make_u32x (MD4C00);
    const u32x F_wac00 = w2[2] + make_u32x (MD4C00);
    const u32x F_wbc00 = w2[3] + make_u32x (MD4C00);
    const u32x F_wcc00 = w3[0] + make_u32x (MD4C00);
    const u32x F_wdc00 = w3[1] + make_u32x (MD4C00);
    const u32x F_wec00 = w3[2] + make_u32x (MD4C00);
    const u32x F_wfc00 = w3[3] + make_u32x (MD4C00);

    const u32x G_w0c01 = w0[0] + make_u32x (MD4C01);
    const u32x G_w4c01 = w1[0] + make_u32x (MD4C01);
    const u32x G_w8c01 = w2[0] + make_u32x (MD4C01);
    const u32x G_wcc01 = w3[0] + make_u32x (MD4C01);
    const u32x G_w1c01 = w0[1] + make_u32x (MD4C01);
    const u32x G_w5c01 = w1[1] + make_u32x (MD4C01);
    const u32x G_w9c01 = w2[1] + make_u32x (MD4C01);
    const u32x G_wdc01 = w3[1] + make_u32x (MD4C01);
    const u32x G_w2c01 = w0[2] + make_u32x (MD4C01);
    const u32x G_w6c01 = w1[2] + make_u32x (MD4C01);
    const u32x G_wac01 = w2[2] + make_u32x (MD4C01);
    const u32x G_wec01 = w3[2] + make_u32x (MD4C01);
    const u32x G_w3c01 = w0[3] + make_u32x (MD4C01);
    const u32x G_w7c01 = w1[3] + make_u32x (MD4C01);
    const u32x G_wbc01 = w2[3] + make_u32x (MD4C01);
    const u32x G_wfc01 = w3[3] + make_u32x (MD4C01);

    const u32x H_w0c02 = w0[0] + make_u32x (MD4C02);
    const u32x H_w8c02 = w2[0] + make_u32x (MD4C02);
    const u32x H_w4c02 = w1[0] + make_u32x (MD4C02);
    const u32x H_wcc02 = w3[0] + make_u32x (MD4C02);
    const u32x H_w2c02 = w0[2] + make_u32x (MD4C02);
    const u32x H_wac02 = w2[2] + make_u32x (MD4C02);
    const u32x H_w6c02 = w1[2] + make_u32x (MD4C02);
    const u32x H_wec02 = w3[2] + make_u32x (MD4C02);
    const u32x H_w1c02 = w0[1] + make_u32x (MD4C02);
    const u32x H_w9c02 = w2[1] + make_u32x (MD4C02);
    const u32x H_w5c02 = w1[1] + make_u32x (MD4C02);
    const u32x H_wdc02 = w3[1] + make_u32x (MD4C02);
    const u32x H_w3c02 = w0[3] + make_u32x (MD4C02);
    const u32x H_wbc02 = w2[3] + make_u32x (MD4C02);
    const u32x H_w7c02 = w1[3] + make_u32x (MD4C02);
    const u32x H_wfc02 = w3[3] + make_u32x (MD4C02);

    /**
     * md4
     */

    u32x a = MD4M_A;
    u32x b = MD4M_B;
    u32x c = MD4M_C;
    u32x d = MD4M_D;

    MD4_STEP0(MD4_Fo, a, b, c, d, F_w0c00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_w1c00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_w2c00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_w3c00, MD4S03);
    MD4_STEP0(MD4_Fo, a, b, c, d, F_w4c00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_w5c00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_w6c00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_w7c00, MD4S03);
    MD4_STEP0(MD4_Fo, a, b, c, d, F_w8c00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_w9c00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_wac00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_wbc00, MD4S03);
    MD4_STEP0(MD4_Fo, a, b, c, d, F_wcc00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_wdc00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_wec00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_wfc00, MD4S03);

    MD4_STEP0(MD4_Go, a, b, c, d, G_w0c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w4c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_w8c01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wcc01, MD4S13);
    MD4_STEP0(MD4_Go, a, b, c, d, G_w1c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w5c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_w9c01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wdc01, MD4S13);
    MD4_STEP0(MD4_Go, a, b, c, d, G_w2c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w6c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_wac01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wec01, MD4S13);
    MD4_STEP0(MD4_Go, a, b, c, d, G_w3c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w7c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_wbc01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wfc01, MD4S13);

    MD4_STEP0(MD4_H , a, b, c, d, H_w0c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_w8c02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w4c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wcc02, MD4S23);
    MD4_STEP0(MD4_H , a, b, c, d, H_w2c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_wac02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w6c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wec02, MD4S23);
    MD4_STEP0(MD4_H , a, b, c, d, H_w1c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_w9c02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w5c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wdc02, MD4S23);
    MD4_STEP0(MD4_H , a, b, c, d, H_w3c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_wbc02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w7c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wfc02, MD4S23);

    COMPARE_M_SIMD (a, d, c, b);
  }
}

KERNEL_FQ KERNEL_FA void m01000_m08 (KERN_ATTR_BASIC ())
{
}

KERNEL_FQ KERNEL_FA void m01000_m16 (KERN_ATTR_BASIC ())
{
}

KERNEL_FQ KERNEL_FA void m01000_s04 (KERN_ATTR_BASIC ())
{
  /**
   * modifier
   */

  const u64 lid = get_local_id (0);

  /**
   * base
   */

  const u64 gid = get_global_id (0);

  if (gid >= GID_CNT) return;

  u32 pw_buf0[4];
  u32 pw_buf1[4];

  pw_buf0[0] = pws[gid].i[0];
  pw_buf0[1] = pws[gid].i[1];
  pw_buf0[2] = pws[gid].i[2];
  pw_buf0[3] = pws[gid].i[3];
  pw_buf1[0] = pws[gid].i[4];
  pw_buf1[1] = pws[gid].i[5];
  pw_buf1[2] = pws[gid].i[6];
  pw_buf1[3] = pws[gid].i[7];

  const u32 pw_l_len = pws[gid].pw_len & 63;

  /**
   * digest
   */

  const u32 search[4] =
  {
    digests_buf[DIGESTS_OFFSET_HOST].digest_buf[DGST_R0],
    digests_buf[DIGESTS_OFFSET_HOST].digest_buf[DGST_R1],
    digests_buf[DIGESTS_OFFSET_HOST].digest_buf[DGST_R2],
    digests_buf[DIGESTS_OFFSET_HOST].digest_buf[DGST_R3]
  };

  /**
   * loop
   */

  for (u32 il_pos = 0; il_pos < IL_CNT; il_pos += VECT_SIZE)
  {
    const u32x pw_r_len = pwlenx_create_combt (combs_buf, il_pos) & 63;

    const u32x pw_len = (pw_l_len + pw_r_len) & 63;

    /**
     * concat password candidate
     */

    u32x wordl0[4] = { 0 };
    u32x wordl1[4] = { 0 };
    u32x wordl2[4] = { 0 };
    u32x wordl3[4] = { 0 };

    wordl0[0] = pw_buf0[0];
    wordl0[1] = pw_buf0[1];
    wordl0[2] = pw_buf0[2];
    wordl0[3] = pw_buf0[3];
    wordl1[0] = pw_buf1[0];
    wordl1[1] = pw_buf1[1];
    wordl1[2] = pw_buf1[2];
    wordl1[3] = pw_buf1[3];

    u32x wordr0[4] = { 0 };
    u32x wordr1[4] = { 0 };
    u32x wordr2[4] = { 0 };
    u32x wordr3[4] = { 0 };

    wordr0[0] = ix_create_combt (combs_buf, il_pos, 0);
    wordr0[1] = ix_create_combt (combs_buf, il_pos, 1);
    wordr0[2] = ix_create_combt (combs_buf, il_pos, 2);
    wordr0[3] = ix_create_combt (combs_buf, il_pos, 3);
    wordr1[0] = ix_create_combt (combs_buf, il_pos, 4);
    wordr1[1] = ix_create_combt (combs_buf, il_pos, 5);
    wordr1[2] = ix_create_combt (combs_buf, il_pos, 6);
    wordr1[3] = ix_create_combt (combs_buf, il_pos, 7);

    if (COMBS_MODE == COMBINATOR_MODE_BASE_LEFT)
    {
      switch_buffer_by_offset_le_VV (wordr0, wordr1, wordr2, wordr3, pw_l_len);
    }
    else
    {
      switch_buffer_by_offset_le_VV (wordl0, wordl1, wordl2, wordl3, pw_r_len);
    }

    u32x w0[4];
    u32x w1[4];
    u32x w2[4];
    u32x w3[4];

    w0[0] = wordl0[0] | wordr0[0];
    w0[1] = wordl0[1] | wordr0[1];
    w0[2] = wordl0[2] | wordr0[2];
    w0[3] = wordl0[3] | wordr0[3];
    w1[0] = wordl1[0] | wordr1[0];
    w1[1] = wordl1[1] | wordr1[1];
    w1[2] = wordl1[2] | wordr1[2];
    w1[3] = wordl1[3] | wordr1[3];
    w2[0] = wordl2[0] | wordr2[0];
    w2[1] = wordl2[1] | wordr2[1];
    w2[2] = wordl2[2] | wordr2[2];
    w2[3] = wordl2[3] | wordr2[3];
    w3[0] = wordl3[0] | wordr3[0];
    w3[1] = wordl3[1] | wordr3[1];
    w3[2] = wordl3[2] | wordr3[2];
    w3[3] = wordl3[3] | wordr3[3];

    make_utf16le (w1, w2, w3);
    make_utf16le (w0, w0, w1);

    w3[2] = pw_len * 8 * 2;
    w3[3] = 0;

    /**
     * precompute w[i] + constant for all 48 MD4 steps
     */

    const u32x F_w0c00 = w0[0] + make_u32x (MD4C00);
    const u32x F_w1c00 = w0[1] + make_u32x (MD4C00);
    const u32x F_w2c00 = w0[2] + make_u32x (MD4C00);
    const u32x F_w3c00 = w0[3] + make_u32x (MD4C00);
    const u32x F_w4c00 = w1[0] + make_u32x (MD4C00);
    const u32x F_w5c00 = w1[1] + make_u32x (MD4C00);
    const u32x F_w6c00 = w1[2] + make_u32x (MD4C00);
    const u32x F_w7c00 = w1[3] + make_u32x (MD4C00);
    const u32x F_w8c00 = w2[0] + make_u32x (MD4C00);
    const u32x F_w9c00 = w2[1] + make_u32x (MD4C00);
    const u32x F_wac00 = w2[2] + make_u32x (MD4C00);
    const u32x F_wbc00 = w2[3] + make_u32x (MD4C00);
    const u32x F_wcc00 = w3[0] + make_u32x (MD4C00);
    const u32x F_wdc00 = w3[1] + make_u32x (MD4C00);
    const u32x F_wec00 = w3[2] + make_u32x (MD4C00);
    const u32x F_wfc00 = w3[3] + make_u32x (MD4C00);

    const u32x G_w0c01 = w0[0] + make_u32x (MD4C01);
    const u32x G_w4c01 = w1[0] + make_u32x (MD4C01);
    const u32x G_w8c01 = w2[0] + make_u32x (MD4C01);
    const u32x G_wcc01 = w3[0] + make_u32x (MD4C01);
    const u32x G_w1c01 = w0[1] + make_u32x (MD4C01);
    const u32x G_w5c01 = w1[1] + make_u32x (MD4C01);
    const u32x G_w9c01 = w2[1] + make_u32x (MD4C01);
    const u32x G_wdc01 = w3[1] + make_u32x (MD4C01);
    const u32x G_w2c01 = w0[2] + make_u32x (MD4C01);
    const u32x G_w6c01 = w1[2] + make_u32x (MD4C01);
    const u32x G_wac01 = w2[2] + make_u32x (MD4C01);
    const u32x G_wec01 = w3[2] + make_u32x (MD4C01);
    const u32x G_w3c01 = w0[3] + make_u32x (MD4C01);
    const u32x G_w7c01 = w1[3] + make_u32x (MD4C01);
    const u32x G_wbc01 = w2[3] + make_u32x (MD4C01);
    const u32x G_wfc01 = w3[3] + make_u32x (MD4C01);

    const u32x H_w0c02 = w0[0] + make_u32x (MD4C02);
    const u32x H_w8c02 = w2[0] + make_u32x (MD4C02);
    const u32x H_w4c02 = w1[0] + make_u32x (MD4C02);
    const u32x H_wcc02 = w3[0] + make_u32x (MD4C02);
    const u32x H_w2c02 = w0[2] + make_u32x (MD4C02);
    const u32x H_wac02 = w2[2] + make_u32x (MD4C02);
    const u32x H_w6c02 = w1[2] + make_u32x (MD4C02);
    const u32x H_wec02 = w3[2] + make_u32x (MD4C02);
    const u32x H_w1c02 = w0[1] + make_u32x (MD4C02);
    const u32x H_w9c02 = w2[1] + make_u32x (MD4C02);
    const u32x H_w5c02 = w1[1] + make_u32x (MD4C02);
    const u32x H_wdc02 = w3[1] + make_u32x (MD4C02);
    const u32x H_w3c02 = w0[3] + make_u32x (MD4C02);
    const u32x H_wbc02 = w2[3] + make_u32x (MD4C02);
    const u32x H_w7c02 = w1[3] + make_u32x (MD4C02);
    const u32x H_wfc02 = w3[3] + make_u32x (MD4C02);

    /**
     * md4
     */

    u32x a = MD4M_A;
    u32x b = MD4M_B;
    u32x c = MD4M_C;
    u32x d = MD4M_D;

    MD4_STEP0(MD4_Fo, a, b, c, d, F_w0c00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_w1c00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_w2c00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_w3c00, MD4S03);
    MD4_STEP0(MD4_Fo, a, b, c, d, F_w4c00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_w5c00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_w6c00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_w7c00, MD4S03);
    MD4_STEP0(MD4_Fo, a, b, c, d, F_w8c00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_w9c00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_wac00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_wbc00, MD4S03);
    MD4_STEP0(MD4_Fo, a, b, c, d, F_wcc00, MD4S00);
    MD4_STEP0(MD4_Fo, d, a, b, c, F_wdc00, MD4S01);
    MD4_STEP0(MD4_Fo, c, d, a, b, F_wec00, MD4S02);
    MD4_STEP0(MD4_Fo, b, c, d, a, F_wfc00, MD4S03);

    MD4_STEP0(MD4_Go, a, b, c, d, G_w0c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w4c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_w8c01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wcc01, MD4S13);
    MD4_STEP0(MD4_Go, a, b, c, d, G_w1c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w5c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_w9c01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wdc01, MD4S13);
    MD4_STEP0(MD4_Go, a, b, c, d, G_w2c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w6c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_wac01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wec01, MD4S13);
    MD4_STEP0(MD4_Go, a, b, c, d, G_w3c01, MD4S10);
    MD4_STEP0(MD4_Go, d, a, b, c, G_w7c01, MD4S11);
    MD4_STEP0(MD4_Go, c, d, a, b, G_wbc01, MD4S12);
    MD4_STEP0(MD4_Go, b, c, d, a, G_wfc01, MD4S13);

    MD4_STEP0(MD4_H , a, b, c, d, H_w0c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_w8c02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w4c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wcc02, MD4S23);
    MD4_STEP0(MD4_H , a, b, c, d, H_w2c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_wac02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w6c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wec02, MD4S23);
    MD4_STEP0(MD4_H , a, b, c, d, H_w1c02, MD4S20);
    MD4_STEP0(MD4_H , d, a, b, c, H_w9c02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w5c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wdc02, MD4S23);
    MD4_STEP0(MD4_H , a, b, c, d, H_w3c02, MD4S20);

    if (MATCHES_NONE_VS (a, search[0])) continue;

    MD4_STEP0(MD4_H , d, a, b, c, H_wbc02, MD4S21);
    MD4_STEP0(MD4_H , c, d, a, b, H_w7c02, MD4S22);
    MD4_STEP0(MD4_H , b, c, d, a, H_wfc02, MD4S23);

    COMPARE_S_SIMD (a, d, c, b);
  }
}

KERNEL_FQ KERNEL_FA void m01000_s08 (KERN_ATTR_BASIC ())
{
}

KERNEL_FQ KERNEL_FA void m01000_s16 (KERN_ATTR_BASIC ())
{
}
