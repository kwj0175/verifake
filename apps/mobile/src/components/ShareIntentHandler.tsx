import React, { useEffect } from 'react';
import { useShareIntent } from 'expo-share-intent';
import { navigationRef } from '../../App';

export const ShareIntentHandler = () => {
    const { hasShareIntent, shareIntent, resetShareIntent } = useShareIntent();

    useEffect(() => {
        if (hasShareIntent && shareIntent) {
            console.log('[ShareIntentHandler] Received share intent:', shareIntent);

            let sharedUrl: string | undefined = undefined;
            let sharedVideoUri: string | undefined = undefined;

            // 1. 텍스트 정보 파싱 (유튜브/웹 링크 등)
            if (shareIntent.text) {
                // 텍스트 내에서 URL 추출
                const urlRegex = /(https?:\/\/[^\s]+)/g;
                const matches = shareIntent.text.match(urlRegex);
                if (matches && matches.length > 0) {
                    sharedUrl = matches[0];
                } else if (shareIntent.text.startsWith('http://') || shareIntent.text.startsWith('https://')) {
                    sharedUrl = shareIntent.text.trim();
                }
            }

            // 2. 파일 정보 파싱 (로컬 동영상 파일 등)
            if (shareIntent.files && shareIntent.files.length > 0) {
                // 비디오 확장자나 mimeType이 비디오인 파일 탐색
                const videoFile = shareIntent.files.find(f => {
                    const mimeType = f.mimeType ? f.mimeType.toLowerCase() : '';
                    const path = f.path ? f.path.toLowerCase() : '';
                    return (
                        mimeType.startsWith('video/') ||
                        path.endsWith('.mp4') ||
                        path.endsWith('.mov') ||
                        path.endsWith('.m4v') ||
                        path.endsWith('.avi') ||
                        path.endsWith('.mkv')
                    );
                });

                if (videoFile) {
                    sharedVideoUri = videoFile.path;
                } else {
                    // 텍스트 형태의 URI가 파일 리스트에 들어오는 케이스 처리
                    const file = shareIntent.files[0];
                    const isUri = file.path.startsWith('http://') || file.path.startsWith('https://');
                    if (isUri || file.mimeType === 'text/uri-list') {
                        sharedUrl = file.path;
                    }
                }
            }

            // 3. 목적지인 DetectionInput 화면으로 이동 및 상태 복구
            if (sharedUrl || sharedVideoUri) {
                console.log(`[ShareIntentHandler] Navigating to DetectionInput with sharedUrl: ${sharedUrl}, sharedVideoUri: ${sharedVideoUri}`);

                const navigateToInput = () => {
                    if (navigationRef.isReady()) {
                        // @ts-ignore
                        navigationRef.navigate('DetectionInput', {
                            sharedUrl,
                            sharedVideoUri
                        });
                        resetShareIntent();
                    } else {
                        // 네비게이션 컨테이너가 로드될 때까지 100ms마다 재시도
                        setTimeout(navigateToInput, 100);
                    }
                };

                navigateToInput();
            } else {
                // 파싱 실패 또는 유효하지 않은 포맷인 경우 리셋 처리하여 앱이 멈추지 않게 방지
                resetShareIntent();
            }
        }
    }, [hasShareIntent, shareIntent]);

    return null;
};
