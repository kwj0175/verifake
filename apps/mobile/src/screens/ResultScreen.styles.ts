import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    //공통 레이아웃
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 20, paddingVertical: 15 },
    backBtn: { padding: 4 },
    headerTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
    scrollBody: { padding: 20, paddingBottom: 100 },
    body: { paddingHorizontal: 20, paddingBottom: 100 },

    //영상 미리보기
    mediaPreview: { width: '100%', height: 200, borderRadius: 20, overflow: 'hidden', backgroundColor: '#161622', marginBottom: 24, position: 'relative', },
    previewImage: { width: '100%', height: '100%' },
    mediaOverlay: { position: 'absolute', top: 16, left: 16 },
    statusBadge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8, },
    statusBadgeText: { color: '#fff', fontSize: 12, fontWeight: '900' },

    // 썸네일 
    thumbnailContainer: { width: '100%', height: 220, borderRadius: 20, overflow: 'hidden', marginBottom: 25 },
    thumbnail: { width: '100%', height: '100%' },
    statusTag: { position: 'absolute', top: 15, right: 15, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
    statusTagText: { color: '#fff', fontWeight: 'bold' },

    // 신뢰도 게이지
    gaugeSection: { marginBottom: 30 },
    gaugeTrack: { height: 10, backgroundColor: '#1e1e2e', borderRadius: 5, overflow: 'hidden' },
    scoreSection: { marginBottom: 24 },
    scoreInfo: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 8 },
    scoreLabel: { color: '#444468', fontSize: 14 },
    scoreValue: { fontSize: 24, fontWeight: 'bold' },
    gaugeBar: { width: '100%', height: 8, backgroundColor: '#1e1e2e', borderRadius: 4 },
    gaugeFill: { height: '100%', borderRadius: 4 },

    // 결과 상단 타이틀
    mainResultTitle: { color: '#fff', fontSize: 22, fontWeight: 'bold', marginBottom: 15 },

    // 이상 점 리스트
    anomalySection: { backgroundColor: '#161622', borderRadius: 20, padding: 20, marginBottom: 25 },
    anomalySubTitle: { color: '#7c6cfa', fontSize: 15, fontWeight: '600', marginBottom: 18 },
    anomalyItem: { marginBottom: 16 },
    anomalyText: { color: '#e1e1e6', fontSize: 14, lineHeight: 24 },

    // 분석 근거 카드
    reasonCard: { backgroundColor: '#161622', borderRadius: 20, padding: 20, marginBottom: 16, borderWidth: 1, borderColor: '#1e1e2e', },
    reasonTitle: { color: '#7c6cfa', fontSize: 13, fontWeight: 'bold', marginBottom: 12 },
    reasonContent: { color: '#e1e1e6', fontSize: 15, lineHeight: 24 },
    pathLabel: { color: '#7c6cfa', fontSize: 12, fontWeight: 'bold', marginTop: 8, marginBottom: 4 },
    pathText: { color: '#e1e1e6', fontSize: 12, lineHeight: 18 },

    // 상세 보고서
    sectionHeader: { marginBottom: 15 },
    detailSectionTitle: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginBottom: 15, marginLeft: 5 },
    chartsRow: { flexDirection: 'row', justifyContent: 'space-around', backgroundColor: '#161622', padding: 20, borderRadius: 24, marginBottom: 25 },
    detailCard: { backgroundColor: '#161622', borderRadius: 20, padding: 20, marginBottom: 16 },
    cardTitle: { color: '#fff', fontSize: 15, fontWeight: 'bold', marginBottom: 10 },
    infoRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
    rowLabel: { color: '#fff', fontSize: 15, marginRight: 8 },
    subLabel: { color: '#7c6cfa', fontSize: 14, fontWeight: 'bold', marginBottom: 8 },
    anomalyListText: { color: '#e1e1e6', fontSize: 13, marginLeft: 8, marginBottom: 4 },
    statsRow: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 15, borderTopWidth: 1, borderTopColor: '#1e1e2e', paddingTop: 10 },
    statItem: { color: '#a0a0ab', fontSize: 12 },

    // 기타 그리드 아이템
    detailGrid: { flexDirection: 'row', gap: 12, marginBottom: 24 },
    detailItem: { flex: 1, backgroundColor: '#11111d', padding: 16, borderRadius: 16, alignItems: 'center' },
    detailLabel: { color: '#444468', fontSize: 12, marginBottom: 4 },
    detailValue: { fontSize: 15, fontWeight: 'bold' },

    // 버튼
    primaryButton: { backgroundColor: '#7c6cfa', height: 56, borderRadius: 16, justifyContent: 'center', alignItems: 'center' },
    primaryButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
    shareBtn: { flexDirection: 'row', backgroundColor: '#1a1a2e', height: 56, borderRadius: 16, alignItems: 'center', justifyContent: 'center', },
    shareBtnText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },

    // 공통 유틸리티
    textFake: { color: '#ff453a', fontWeight: 'bold' },
    textReal: { color: '#32d74b', fontWeight: 'bold' },
    bgFake: { backgroundColor: '#ff453a' },
    bgReal: { backgroundColor: '#32d74b' },
    limitCard: { backgroundColor: '#1a1a2e', borderRadius: 16, padding: 18, borderWidth: 1, borderColor: '#ff9f0a33' },
    limitTitle: { color: '#ff9f0a', fontWeight: 'bold', marginBottom: 6, fontSize: 14 },
    limitContent: { color: '#a0a0ab', fontSize: 13, lineHeight: 20 },
});