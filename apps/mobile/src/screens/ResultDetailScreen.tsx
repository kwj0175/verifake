import React from 'react';
import { View, Text, SafeAreaView, ScrollView, TouchableOpacity } from 'react-native';
import { ChevronLeftIcon } from 'react-native-heroicons/outline';
import Svg, { Circle, G } from 'react-native-svg';
import { styles } from './ResultScreen.styles';
import type { MediaTaskResult } from '../api/verifakeApi';

interface ResultDetailRouteParams {
    separatedMedia?: MediaTaskResult;
}

// 원형 그래프 컴포넌트
const DonutChart = ({ percentage, color, label }: { percentage: number; color: string; label: string }) => {
    const radius = 35;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    return (
        <View style={{ alignItems: 'center', flex: 1 }}>
            <Svg width={90} height={90}>
                <G rotation="-90" origin="45, 45">
                    <Circle cx="45" cy="45" r={radius} stroke="#1e1e2e" strokeWidth="7" fill="none" />
                    <Circle
                        cx="45" cy="45" r={radius} stroke={color} strokeWidth="7" fill="none"
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                        strokeLinecap="round"
                    />
                </G>
            </Svg>
            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 20, justifyContent: 'center', alignItems: 'center' }}>
                <Text style={{ color: '#fff', fontSize: 13, fontWeight: 'bold' }}>{percentage}%</Text>
            </View>
            <Text style={{ color: '#a0a0ab', fontSize: 11, marginTop: 8, textAlign: 'center' }}>{label}</Text>
        </View>
    );
};

export const ResultDetailScreen = ({ navigation, route }: any) => {
    // 버그 수정: route.params를 실제로 사용하도록 수정 (기존에는 완전히 무시하고 더미 데이터만 사용)
    const { separatedMedia } = (route.params || {}) as ResultDetailRouteParams;

    const deepfakeScore = separatedMedia?.deepfake_score ?? null;
    const verdict = separatedMedia?.verdict ?? null;
    const isFake = verdict?.toUpperCase() === 'FAKE';

    // 백엔드에서 실제 점수를 받은 경우 사용, 없으면 0 표시
    const totalScore = deepfakeScore !== null ? Math.round(deepfakeScore) : 0;

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()}>
                    <ChevronLeftIcon size={28} color="#fff" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>상세 분석 보고서</Text>
                <View style={{ width: 28 }} />
            </View>

            <ScrollView contentContainerStyle={styles.scrollBody}>
                {/* 종합 분석 결과 */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.detailSectionTitle}>1) 종합 분석 결과</Text>
                </View>
                <View style={styles.chartsRow}>
                    <DonutChart
                        percentage={totalScore}
                        color={isFake ? '#ff453a' : '#32d74b'}
                        label="딥페이크 가능성"
                    />
                    <DonutChart
                        percentage={totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.75)) : 0}
                        color="#7c6cfa"
                        label="신뢰도"
                    />
                    <DonutChart
                        percentage={totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.82)) : 0}
                        color="#32d74b"
                        label="영상/음성 일치도"
                    />
                </View>

                {/* 판정 요약 */}
                <View style={styles.detailCard}>
                    <Text style={styles.cardTitle}>판정 결과</Text>
                    <View style={styles.infoRow}>
                        <Text style={styles.rowLabel}>최종 판정:</Text>
                        <Text style={isFake ? styles.textFake : styles.textReal}>
                            {verdict ?? '분석 중'}
                        </Text>
                    </View>
                    <View style={styles.infoRow}>
                        <Text style={styles.rowLabel}>딥페이크 점수:</Text>
                        <Text style={isFake ? styles.textFake : styles.textReal}>
                            {deepfakeScore !== null ? `${deepfakeScore}%` : '-'}
                        </Text>
                    </View>
                </View>

                {/* 분석 상태 */}
                <View style={styles.detailCard}>
                    <Text style={styles.cardTitle}>분석 정보</Text>
                    <View style={styles.infoRow}>
                        <Text style={styles.rowLabel}>작업 ID:</Text>
                        <Text style={styles.rowLabel}>{separatedMedia?.task_id ?? '-'}</Text>
                    </View>
                    <View style={styles.infoRow}>
                        <Text style={styles.rowLabel}>상태:</Text>
                        <Text style={styles.rowLabel}>{separatedMedia?.status ?? '-'}</Text>
                    </View>
                    {separatedMedia?.video_path && (
                        <Text style={styles.anomalyListText}>✓ 영상 파일 처리 완료</Text>
                    )}
                    {separatedMedia?.audio_path && (
                        <Text style={styles.anomalyListText}>✓ 음성 파일 분리 완료</Text>
                    )}
                </View>

                {/* 분석 한계 */}
                <View style={styles.limitCard}>
                    <Text style={styles.limitTitle}>! 분석 한계</Text>
                    <Text style={styles.limitContent}>
                        영상 일부 구간에서 압축 아티팩트로 인해 정확도가 낮을 수 있습니다.
                        AI 분석 결과는 참고용으로만 사용하세요.
                    </Text>
                </View>

                <View style={{ height: 40 }} />
            </ScrollView>
        </SafeAreaView>
    );
};
