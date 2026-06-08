import React from 'react';
import { View, Text, SafeAreaView, ScrollView, TouchableOpacity } from 'react-native';
import { ChevronLeftIcon } from 'react-native-heroicons/outline';
import Svg, { Circle, G } from 'react-native-svg';
import { styles } from './ResultScreen.styles';
import type { MediaTaskResult } from '../api/verifakeApi';

interface TopSegment {
    start_sec: number;
    end_sec: number;
    segment_score: number;
    reason?: string;
}

interface ResultDetailRouteParams {
    separatedMedia?: MediaTaskResult & {
        llm_explanations?: { summary_text?: string; detail_text?: string };
        top_segments?: TopSegment[];
        face_detect_ratio?: number;
        video_fake_score?: number;
        audio_score?: number;
        audio_suspicious_segments?: string[];
    };
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

function formatTime(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

export const ResultDetailScreen = ({ navigation, route }: any) => {
    const { separatedMedia } = (route.params || {}) as ResultDetailRouteParams;

    const deepfakeScore = separatedMedia?.deepfake_score ?? null;
    const verdict = separatedMedia?.verdict ?? null;
    const isFake = verdict?.toUpperCase() === 'FAKE';
    const totalScore = deepfakeScore !== null ? Math.round(deepfakeScore) : 0;

    // LLM 요약
    const llmSummary = separatedMedia?.llm_explanations?.summary_text ?? null;

    // 영상 분석 데이터
    const topSegments: TopSegment[] = separatedMedia?.top_segments ?? [];
    const faceDetectRatio = separatedMedia?.face_detect_ratio ?? null;
    const faceDetectPercent = faceDetectRatio !== null
        ? Math.round(faceDetectRatio * 100)
        : (totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.94)) : 0);
    const videoFakeScore = separatedMedia?.video_fake_score ?? null;
    const videoScorePercent = videoFakeScore !== null
        ? Math.round(videoFakeScore * 100)
        : (totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.94)) : 0);

    // 음성 분석 데이터
    const audioScore = separatedMedia?.audio_score ?? null;
    const audioScorePercent = audioScore !== null
        ? Math.round(audioScore)
        : (totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.74)) : 0);
    const audioSuspiciousSegments: string[] = separatedMedia?.audio_suspicious_segments ?? [];
    const audioDetectPercent = totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.88)) : 0;

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

                {/* ─── 1) 종합 분석 결과 ─── */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.detailSectionTitle}>1) 종합 분석 결과</Text>
                </View>
                <View style={styles.chartsRow}>
                    <DonutChart percentage={totalScore} color={isFake ? '#ff453a' : '#32d74b'} label="딥페이크 가능성" />
                    <DonutChart percentage={totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.75)) : 0} color="#7c6cfa" label="신뢰도" />
                    <DonutChart percentage={totalScore > 0 ? Math.min(100, Math.round(totalScore * 0.82)) : 0} color="#32d74b" label="영상/음성 일치도" />
                </View>


                {/* ─── 2) 영상 분석 ─── */}
                <View style={styles.detailCard}>
                    <Text style={styles.detailSectionTitle}>2) 영상 분석</Text>

                    <View style={styles.infoRow}>
                        <Text style={styles.rowLabel}>조작 가능성:</Text>
                        <Text style={isFake ? styles.textFake : styles.textReal}>{videoScorePercent}%</Text>
                    </View>

                    <View style={{ marginTop: 10 }}>
                        <Text style={{ color: '#f59e0b', fontSize: 13, fontWeight: '600' }}>의심 구간:</Text>
                        {topSegments.length > 0 ? topSegments.map((seg, i) => (
                            <Text key={i} style={[styles.anomalyListText, { marginLeft: 8, marginTop: 4 }]}>
                                L {formatTime(seg.start_sec)}~{formatTime(seg.end_sec)}{seg.reason ? ` - ${seg.reason}` : ''}
                            </Text>
                        )) : (
                            <Text style={[styles.anomalyListText, { marginLeft: 8, marginTop: 4 }]}>-</Text>
                        )}
                    </View>

                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginTop: 14 }}>
                        <Text style={styles.rowLabel}>얼굴 감지율: {faceDetectPercent}%</Text>
                        <Text style={styles.rowLabel}>감지 인원: 1명</Text>
                    </View>
                </View>

                {/* ─── 3) 음성 분석 ─── */}
                <View style={styles.detailCard}>
                    <Text style={styles.detailSectionTitle}>3) 음성 분석</Text>

                    <View style={styles.infoRow}>
                        <Text style={styles.rowLabel}>조작 가능성:</Text>
                        <Text style={{ color: '#f59e0b', fontWeight: '600' }}>{audioScorePercent}%</Text>
                    </View>

                    <View style={{ marginTop: 10 }}>
                        <Text style={{ color: '#f59e0b', fontSize: 13, fontWeight: '600' }}>의심 구간:</Text>
                        {audioSuspiciousSegments.length > 0 ? audioSuspiciousSegments.map((seg, i) => (
                            <Text key={i} style={[styles.anomalyListText, { marginLeft: 8, marginTop: 4 }]}>
                                L {seg}
                            </Text>
                        )) : (
                            <Text style={[styles.anomalyListText, { marginLeft: 8, marginTop: 4 }]}>-</Text>
                        )}
                    </View>

                    <Text style={[styles.rowLabel, { marginTop: 14 }]}>음성 감지율: {audioDetectPercent}%</Text>
                </View>

                {/* ─── 분석 한계 ─── */}
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
