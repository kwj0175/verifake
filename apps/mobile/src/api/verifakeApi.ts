const DEFAULT_API_BASE_URL = 'http://localhost:8000';
const API_BASE_URL = (process.env.EXPO_PUBLIC_API_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, '');
const DEFAULT_POLL_ATTEMPTS = 120;
const DEFAULT_POLL_INTERVAL_MS = 2000;

// 버그 수정: 백엔드 실제 status 값에 맞게 수정
// 'DONE' → 'COMPLETED', 'PREPROCESSING'/'ANALYZING' 추가
export type MediaTaskStatus =
    | 'PENDING'
    | 'DOWNLOADING'
    | 'PROCESSING'
    | 'PREPROCESSING'
    | 'ANALYZING'
    | 'COMPLETED'
    | 'FAILED';

export interface MediaTaskResponse {
    task_id: string;
    timestamp: string;
    message: string;
}

export interface MediaTaskResult {
    task_id: string;
    status: MediaTaskStatus;
    verdict: string | null;
    // 버그 수정: 백엔드 응답 키 deepfake_score와 일치하도록 수정
    deepfake_score?: number | null;
    video_path?: string | null;
    audio_path?: string | null;
    error?: string | null;
    created_at?: string | null;
}

// 버그 수정: 백엔드 /api/v1/history 응답 구조 {total, items, limit, offset}에 맞게 타입 추가
export interface HistoryListResponse {
    total: number;
    items: HistoryItemRaw[];
    limit: number;
    offset: number;
}

// 백엔드 VideoMetadata 필드명과 일치하는 원본 타입
export interface HistoryItemRaw {
    id: number;          // 버그 수정: 백엔드는 int 반환
    task_id: string;
    status: string;
    verdict: string | null;
    deepfake_score: number | null;  // 버그 수정: 백엔드 필드명
    title: string | null;
    thumbnail_path: string | null;
    created_at: string | null;      // 버그 수정: 백엔드 필드명
}

// 화면에서 사용하는 가공된 타입
export interface HistoryItem {
    id: string;
    task_id: string;
    title: string;
    status: string;
    verdict: string | null;
    score: number | null;
    date: string;
    thumb?: string | null;
}

// 버그 수정: 백엔드 응답 구조에서 items 배열을 꺼내고 필드명 매핑
export async function getHistory(userId?: string): Promise<HistoryItem[]> {
    const url = userId
        ? `${API_BASE_URL}/api/v1/history?user_id=${encodeURIComponent(userId)}`
        : `${API_BASE_URL}/api/v1/history`;
    const response = await fetch(url);
    const data = await parseJsonResponse<HistoryListResponse>(response);
    return data.items.map((raw) => ({
        id: String(raw.id),
        task_id: raw.task_id,
        title: raw.title ?? '분석 영상',
        status: raw.status,
        verdict: raw.verdict,
        score: raw.deepfake_score,
        date: raw.created_at ?? '',
        thumb: raw.thumbnail_path,
    }));
}

class ApiError extends Error {
    constructor(message: string, readonly status: number) {
        super(message);
        this.name = 'ApiError';
    }
}

function getErrorMessage(payload: unknown, fallback: string): string {
    if (typeof payload !== 'object' || payload === null) {
        return fallback;
    }
    const detail = (payload as Record<string, unknown>).detail;
    if (typeof detail === 'string') {
        return detail;
    }
    const message = (payload as Record<string, unknown>).message;
    if (typeof message === 'string') {
        return message;
    }
    return fallback;
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) {
        throw new ApiError(getErrorMessage(payload, `HTTP ${response.status}`), response.status);
    }
    return payload as T;
}

function getFilenameFromUri(uri: string): string {
    const withoutQuery = uri.split('?')[0] ?? uri;
    const filename = withoutQuery.split('/').filter(Boolean).at(-1);
    return filename && filename.includes('.') ? filename : 'upload.mp4';
}

async function createVideoUploadBody(uri: string, title: string): Promise<FormData> {
    const fileResponse = await fetch(uri);
    if (!fileResponse.ok) {
        throw new ApiError('선택한 영상을 읽을 수 없습니다.', fileResponse.status);
    }
    const blob = await fileResponse.blob();
    const body = new FormData();
    body.append('title', title);
    body.append('videoFile', blob, getFilenameFromUri(uri));
    return body;
}

export async function uploadVideoForSeparation(uri: string, title = 'mobile-upload'): Promise<MediaTaskResponse> {
    const body = await createVideoUploadBody(uri, title);
    const response = await fetch(`${API_BASE_URL}/api/v1/video`, {
        method: 'POST',
        body,
    });
    return parseJsonResponse<MediaTaskResponse>(response);
}

export async function collectInstagramVideo(link: string, title = 'mobile-link'): Promise<MediaTaskResponse> {
    const body = new FormData();
    body.append('title', title);
    body.append('link', link);
    const response = await fetch(`${API_BASE_URL}/api/v1/instagram`, {
        method: 'POST',
        body,
    });
    return parseJsonResponse<MediaTaskResponse>(response);
}

export async function getMediaTaskStatus(taskId: string): Promise<MediaTaskResult> {
    const response = await fetch(`${API_BASE_URL}/api/v1/status/${encodeURIComponent(taskId)}`);
    return parseJsonResponse<MediaTaskResult>(response);
}

function delay(milliseconds: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

export async function waitForMediaSeparation(
    taskId: string,
    maxAttempts = DEFAULT_POLL_ATTEMPTS,
    intervalMs = DEFAULT_POLL_INTERVAL_MS,
): Promise<MediaTaskResult> {
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        const result = await getMediaTaskStatus(taskId);

        // 버그 수정: 'DONE' → 'COMPLETED'
        if (result.status === 'COMPLETED') {
            return result;
        }

        if (result.status === 'FAILED') {
            throw new ApiError(result.error ?? '영상/음성 분리에 실패했습니다.', 500);
        }

        await delay(intervalMs);
    }

    throw new ApiError('영상/음성 분리 결과를 기다리는 시간이 초과되었습니다.', 408);
}
