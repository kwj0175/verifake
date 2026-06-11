import React, { useEffect, useState } from 'react';
import { View, Text, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { styles, iconProps } from './AnalysisScreen.styles';
import { ClipboardDocumentCheckIcon, CheckIcon, EllipsisHorizontalIcon } from 'react-native-heroicons/outline';
import { BottomNavigation } from '../components/BottomNavigaton';
import {
    uploadVideoForSeparation,
    collectInstagramVideo,
    getMediaTaskStatus,
} from '../api/verifakeApi';

type StepStatus = 'done' | 'loading' | 'wait';

const STEPS = [
    '영상을 서버로 보내는 중',
    '영상 파일 불러오는 중',
    'AI가 영상을 분석하는 중',
    '분석 완료!',
];

function getScreenText(activeStep: number): { title: string; subTitle: string } {
    switch (activeStep) {
        case 1: return { title: '영상을 보내는 중이에요', subTitle: '잠깐만 기다려 주세요' };
        case 2: return { title: '영상을 불러오는 중이에요', subTitle: '서버에서 영상을 준비하고 있어요' };
        case 3: return { title: 'AI가 분석하고 있어요', subTitle: '딥페이크 여부를 꼼꼼히 살펴보고 있어요' };
        case 4: return { title: '분석이 끝났어요!', subTitle: '결과를 확인해 보세요' };
        default: return { title: '분석 중이에요', subTitle: '잠깐만 기다려 주세요' };
    }
}

function getStepStatus(stepNumber: number, activeStep: number): StepStatus {
    if (stepNumber < activeStep) return 'done';
    if (stepNumber === activeStep) return 'loading';
    return 'wait';
}

const MAX_POLL_ATTEMPTS = 300;
const POLL_INTERVAL_MS = 2000;

export const AnalysisScreen = ({ navigation, route }: any) => {
    const [activeStep, setActiveStep] = useState(1);
    const { thumbnailUri, videoUri, url } = route.params || {};
    const { title, subTitle } = getScreenText(activeStep);

    useEffect(() => {
        let isMounted = true;

        async function runAnalysis() {
            try {
                // Step 1: 영상 업로드
                setActiveStep(1);
                let taskId: string;

                if (videoUri) {
                    const res = await uploadVideoForSeparation(videoUri);
                    taskId = res.task_id;
                } else if (url) {
                    const res = await collectInstagramVideo(url);
                    taskId = res.task_id;
                } else {
                    Alert.alert('오류', '분석할 영상이 없습니다.');
                    navigation.goBack();
                    return;
                }

                // Steps 2-4: 상태 폴링
                for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
                    if (!isMounted) return;

                    const result = await getMediaTaskStatus(taskId);

                    // 버그 수정: 백엔드 실제 status 값에 맞게 분기 수정
                    if (result.status === 'PENDING' || result.status === 'DOWNLOADING') {
                        setActiveStep(2);
                    } else if (
                        result.status === 'PROCESSING' ||
                        result.status === 'PREPROCESSING'  // 버그 수정: PREPROCESSING 상태 추가
                    ) {
                        setActiveStep(2);
                    } else if (result.status === 'ANALYZING') {
                        // 버그 수정: ANALYZING 상태 → step 3 (AI 분석 중)
                        setActiveStep(3);
                    } else if (result.status === 'COMPLETED') {
                        // 버그 수정: 'DONE' → 'COMPLETED'
                        setActiveStep(4);
                        await new Promise((r) => setTimeout(r, 600));
                        if (isMounted) {
                            navigation.navigate('Result', { thumbnailUri, separatedMedia: result });
                        }
                        return;
                    } else if (result.status === 'FAILED') {
                        Alert.alert('분석 실패', result.error ?? '분석에 실패했습니다.');
                        navigation.goBack();
                        return;
                    }

                    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
                }

                if (isMounted) {
                    Alert.alert('시간 초과', '분석 결과를 기다리는 시간이 초과되었습니다.');
                    navigation.goBack();
                }
            } catch (e: any) {
                if (!isMounted) return;
                Alert.alert('오류', e.message ?? '알 수 없는 오류가 발생했습니다.');
                navigation.goBack();
            }
        }

        runAnalysis();

        return () => {
            isMounted = false;
        };
    }, []);

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.content}>
                <View style={styles.loaderContainer}>
                    <View style={styles.outerCircle}>
                        <View style={styles.innerCircle}>
                            <ClipboardDocumentCheckIcon size={40} color="#7c6cfa" strokeWidth={2} />
                        </View>
                    </View>
                    <Text style={styles.title}>{title}</Text>
                    <Text style={styles.subTitle}>{subTitle}</Text>
                </View>

                <View style={styles.stepList}>
                    {STEPS.map((label, index) => {
                        const stepNumber = index + 1;
                        const status = getStepStatus(stepNumber, activeStep);
                        return (
                            <View key={stepNumber} style={[styles.stepItem, status === 'done' ? styles.stepDone : styles.stepWait]}>
                                <View style={[styles.checkCircle, status === 'done' && styles.checkCircleDone]}>
                                    {status === 'done' && <CheckIcon {...iconProps.checkIcon} />}
                                    {status === 'loading' && <EllipsisHorizontalIcon {...iconProps.loadingIcon} />}
                                </View>
                                <Text style={[styles.stepLabel, status === 'done' && styles.textDone]}>
                                    {label}
                                </Text>
                            </View>
                        );
                    })}
                </View>
            </View>

            <BottomNavigation navigation={navigation} activeRoute="DetectionInput" />
        </SafeAreaView>
    );
};
