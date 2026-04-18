      SUBROUTINE LEGACY_PREPARE_GRADIENT_MODULES(VB_STR)
!
!     Populate the legacy RDM/orbital-gradient modules from the current
!     `vb_info` buffers without entering the full legacy optimizer. This is
!     the minimal state preparation needed before calling `orbprep`,
!     `cal_g11`, `grad_rdm5`, and `grdori` on a standalone runtime snapshot.
!
      USE, INTRINSIC :: ISO_C_BINDING
      USE :: VB_MOD
      USE :: DVNV
      USE :: SYSPARAM
      USE :: COEF
      USE :: DETOV
      USE :: EDATA
      USE :: HSG
      USE :: HSGF
      USE :: HSGO
      USE :: INDXP
      USE :: IOR
      USE :: LCNM4
      USE :: ORB
      USE :: ORBIDX
      USE :: ORBTRAN
      USE :: PTORB
      USE :: STRHAMOV
      USE :: SYSPARAM_AUX
      USE :: TOTSTR
      USE :: NSFVP
      USE :: NTHD
      USE :: VBXMPAR
      IMPLICIT NONE

      TYPE(VB_INFO) :: VB_STR
      REAL(C_DOUBLE), POINTER :: COL_TMP(:), DV_TMP(:,:)
      REAL(C_DOUBLE), POINTER :: SSF_TMP(:,:), HHF_TMP(:,:), GGF_TMP(:)
      INTEGER(C_INT), POINTER :: NTSTR_TMP(:,:)
      INTEGER(C_INT), POINTER :: NV_TMP(:,:), MA_TMP(:)
      INTEGER(C_INT), POINTER :: INDXXC_TMP(:), INDXCX_TMP(:,:)
      INTEGER(C_INT), POINTER :: IOOR_TMP(:), INOR_TMP(:)
      INTEGER :: I,J,NSAV

      WRITE(0,*) 'LEGACY_PREPARE_GRADIENT_MODULES stage=begin'
      CALL INIT_RDM_MODULES(VB_STR)
      WRITE(0,*) 'LEGACY_PREPARE_GRADIENT_MODULES stage=after_init_rdm_modules'

      HES_IW = VB_STR%IW
      HES_IERR = VB_STR%IER
      ENUCL = VB_STR%ENUC
      GPG = VB_STR%GPG
      IOPT = 6
      ITMAX = VB_STR%ITMAX

      NAO = VB_STR%NAO
      NAE = VB_STR%NAE
      NCOR = VB_STR%NCOR
      NAO2 = (NAO-1)*NAO/2
      NAO2C = (NAO+1)*NAO/2
      NAO2B = (NAO-1)*NAO/2
      NAO3 = NAO*NAO*(NAO+1)/2
      NAO3B = (NAO-2)*NAO2/3
      NAO4 = NAO2C*(NAO2C+1)/2
      NAO4B = (NAO-3)*NAO3B/4

      NOR_MAX = VB_STR%NB
      NB = VB_STR%NB
      NB2 = (NB+1)*NB/2
      NB4 = NB2*(NB2+1)/2

      NOR = VB_STR%NOR
      NVAR = VB_STR%NVAR
      NSTR = VB_STR%NSTR
      NSAV = VB_STR%NSAV
      NEL = VB_STR%NEL
      MXBOND = VB_STR%MXBOND
      NMUL = VB_STR%NMUL
      IVBXM = VB_STR%WFNTYP + 1

      IF (.NOT.ALLOCATED(NTSTR)) ALLOCATE(NTSTR(NEL,NSTR))
      CALL C_F_POINTER(VB_STR%NTSTR, NTSTR_TMP, [NEL,NSTR])
      DO I=1,NEL
        DO J=1,NSTR
          NTSTR(I,J) = NTSTR_TMP(I,J)
        ENDDO
      ENDDO

      IF (.NOT.ALLOCATED(COL)) ALLOCATE(COL(NSTR*NSAV))
      CALL C_F_POINTER(VB_STR%COL, COL_TMP, [NSTR*NSAV])
      DO I=1,NSTR*NSAV
        COL(I) = COL_TMP(I)
      ENDDO
      WRITE(0,*) 'LEGACY_PREPARE_GRADIENT_MODULES stage=after_col_copy'

      CALL C_F_POINTER(VB_STR%DV, DV_TMP, [NB,NB])
      CALL C_F_POINTER(VB_STR%NV, NV_TMP, [NB,NOR])
      CALL C_F_POINTER(VB_STR%MA, MA_TMP, [NOR])

      IF (.NOT.ALLOCATED(DV)) ALLOCATE(DV(NB,NB))
      IF (.NOT.ALLOCATED(NV)) ALLOCATE(NV(NB,NOR))
      IF (.NOT.ALLOCATED(MA)) ALLOCATE(MA(NOR))
      DV = 0.D0
      NV = 0
      MA = 0
      DV(1:NB,1:NB) = DV_TMP(1:NB,1:NB)
      NV(1:NB,1:NOR) = NV_TMP(1:NB,1:NOR)
      MA(1:NOR) = MA_TMP(1:NOR)

      CALL C_F_POINTER(VB_STR%SSF, SSF_TMP, [NB,NB])
      CALL C_F_POINTER(VB_STR%HHF, HHF_TMP, [NB,NB])
      CALL C_F_POINTER(VB_STR%GGF, GGF_TMP, [VB_STR%N2E])

      IF (.NOT.ALLOCATED(SSF)) ALLOCATE(SSF(NB,NB))
      IF (.NOT.ALLOCATED(HHF)) ALLOCATE(HHF(NB,NB))
      IF (.NOT.ALLOCATED(GGF)) ALLOCATE(GGF(NB4))
      SSF = 0.D0
      HHF = 0.D0
      GGF = 0.D0
      SSF(1:NB,1:NB) = SSF_TMP(1:NB,1:NB)
      HHF(1:NB,1:NB) = HHF_TMP(1:NB,1:NB)
      GGF(1:NB4) = GGF_TMP(1:NB4)
      WRITE(0,*) 'LEGACY_PREPARE_GRADIENT_MODULES stage=after_hsgf_copy'

      IF (.NOT.ALLOCATED(HHO)) ALLOCATE(HHO(NB,NB))
      IF (.NOT.ALLOCATED(SSO)) ALLOCATE(SSO(NB,NB))
      IF (.NOT.ALLOCATED(GGO)) ALLOCATE(GGO(NAO4))
      IF (.NOT.ALLOCATED(HH)) ALLOCATE(HH(NOR_MAX,NOR_MAX))
      IF (.NOT.ALLOCATED(SS)) ALLOCATE(SS(NOR_MAX,NOR_MAX))
      HHO = 0.D0
      SSO = 0.D0
      GGO = 0.D0
      HH = 0.D0
      SS = 0.D0

      CALL C_F_POINTER(VB_STR%INDXXC, INDXXC_TMP, [2*NVAR])
      CALL C_F_POINTER(VB_STR%INDXCX, INDXCX_TMP, [NB,NOR])
      CALL C_F_POINTER(VB_STR%IOOR, IOOR_TMP, [NOR])
      CALL C_F_POINTER(VB_STR%INOR, INOR_TMP, [NOR])

      IF (.NOT.ALLOCATED(INDXXC)) ALLOCATE(INDXXC(2*NVAR))
      IF (.NOT.ALLOCATED(INDXCX)) ALLOCATE(INDXCX(NB,NOR))
      IF (.NOT.ALLOCATED(IOOR)) ALLOCATE(IOOR(NOR))
      IF (.NOT.ALLOCATED(INOR)) ALLOCATE(INOR(NOR))
      DO I=1,2*NVAR
        INDXXC(I) = INDXXC_TMP(I) + 1
      ENDDO
      DO I=1,NB
        DO J=1,NOR
          INDXCX(I,J) = INDXCX_TMP(I,J) + 1
        ENDDO
      ENDDO
      DO I=1,NOR
        IOOR(I) = IOOR_TMP(I) + 1
        INOR(I) = INOR_TMP(I) + 1
      ENDDO
      WRITE(0,*) 'LEGACY_PREPARE_GRADIENT_MODULES stage=after_index_copy'

      IF (.NOT.ALLOCATED(LCNMO4)) ALLOCATE(LCNMO4(NAO,NAO,NAO,NAO))
      IF (.NOT.ALLOCATED(MAX)) ALLOCATE(MAX(NOR_MAX))
      IF (.NOT.ALLOCATED(NVIC)) ALLOCATE(NVIC(NB,NOR_MAX))
      IF (.NOT.ALLOCATED(OOTran)) ALLOCATE(OOTran(NB,NB))
      IF (.NOT.ALLOCATED(OBTran)) ALLOCATE(OBTran(NB,NB))
      IF (.NOT.ALLOCATED(TAUX)) ALLOCATE(TAUX(NB,NB))
      IF (.NOT.ALLOCATED(TORI)) ALLOCATE(TORI(NB,NB))
      IF (.NOT.ALLOCATED(OVDET)) ALLOCATE(OVDET(NHDTOT))
      IF (.NOT.ALLOCATED(TPT)) ALLOCATE(TPT(NB,NB))
      IF (.NOT.ALLOCATED(HVB)) ALLOCATE(HVB(NSTR,NSTR))
      IF (.NOT.ALLOCATED(SVB)) ALLOCATE(SVB(NSTR,NSTR))
      MAX = 0
      NVIC = 0
      OOTran = 0.D0
      OBTran = 0.D0
      TAUX = 0.D0
      TORI = 0.D0
      OVDET = 0.D0
      TPT = 0.D0
      HVB = 0.D0
      SVB = 0.D0
      WRITE(0,*) 'LEGACY_PREPARE_GRADIENT_MODULES stage=end'

      RETURN
      END SUBROUTINE LEGACY_PREPARE_GRADIENT_MODULES

      SUBROUTINE LEGACY_SYNC_STRUCTURE_RESULTS(VB_STR)
!
!     Copy the legacy structure/eigenvector work buffers back to the C-side
!     `vb_info` storage after `hamhd`/`hamov`/`eigencalc`.
!
      USE, INTRINSIC :: ISO_C_BINDING
      USE :: VB_MOD
      USE :: COEF
      USE :: STRHAMOV
      IMPLICIT NONE

      TYPE(VB_INFO) :: VB_STR
      REAL(C_DOUBLE), POINTER :: COL_TMP(:)
      REAL(C_DOUBLE), POINTER :: HVB_TMP(:,:), SVB_TMP(:,:)
      INTEGER :: NSTR, NSAV

      NSTR = VB_STR%NSTR
      NSAV = VB_STR%NSAV
      CALL C_F_POINTER(VB_STR%COL, COL_TMP, [NSTR*NSAV])
      CALL C_F_POINTER(VB_STR%HVB, HVB_TMP, [NSTR,NSTR])
      CALL C_F_POINTER(VB_STR%SVB, SVB_TMP, [NSTR,NSTR])

      COL_TMP(1:NSTR*NSAV) = COL(1:NSTR*NSAV)
      HVB_TMP(1:NSTR,1:NSTR) = HVB(1:NSTR,1:NSTR)
      SVB_TMP(1:NSTR,1:NSTR) = SVB(1:NSTR,1:NSTR)

      RETURN
      END SUBROUTINE LEGACY_SYNC_STRUCTURE_RESULTS

      SUBROUTINE LEGACY_BUILD_GRDAUX(GRDA,GRDV,GRDAUX,VB_STR)
!
!     Build the legacy auxiliary active-orbital coefficient gradient
!     `Grdaux(J,I)` from the state-specific `Grda` / `Grdv` layers and the
!     current `A5 = Taux^{-1}` cache stored in module `ORBGRAD`.
!
!     Dimensions:
!       GRDA   : (Nao, Nao)
!       GRDV   : (Nb,  Nb)
!       GRDAUX : (Nb,  Nao)
!
      USE ORBGRAD
      USE :: VB_MOD
      IMPLICIT NONE

      TYPE(VB_INFO) :: VB_STR
      DOUBLE PRECISION :: GRDA(VB_STR%NAO,VB_STR%NAO)
      DOUBLE PRECISION :: GRDV(VB_STR%NB,VB_STR%NB)
      DOUBLE PRECISION :: GRDAUX(VB_STR%NB,VB_STR%NAO)
      INTEGER :: I,J,K
      INTEGER :: NDB,NOR,NVIR,NB,NAO

      NB = VB_STR%NB
      NAO = VB_STR%NAO
      NOR = VB_STR%NOR
      NDB = NOR - NAO
      NVIR = NB - NOR

      GRDAUX = 0.D0

      DO I=1,NAO
        DO J=1,NB
          DO K=1,NAO
            GRDAUX(J,I)=GRDAUX(J,I)+GRDA(K,I)*A5(K+NDB,J)
          ENDDO
          DO K=1,NVIR
            GRDAUX(J,I)=GRDAUX(J,I)+GRDV(K,I)*A5(K+NOR,J)
          ENDDO
        ENDDO
      ENDDO

      RETURN
      END SUBROUTINE LEGACY_BUILD_GRDAUX
