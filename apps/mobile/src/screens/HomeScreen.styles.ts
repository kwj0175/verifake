import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    header: { paddingHorizontal: 20, paddingVertical: 16 },
    logo: { color: '#7c6cfa', fontSize: 20, fontWeight: 'bold' },
    body: { paddingHorizontal: 20, paddingTop: 40, paddingBottom: 100 },

    introSection: { marginBottom: 48 },
    mainTitle: { color: '#fff', fontSize: 28, fontWeight: 'bold', lineHeight: 38 },
    subTitle: { color: '#444468', fontSize: 15, marginTop: 12, lineHeight: 22 },

    mainBtn: {
        backgroundColor: '#7c6cfa',
        borderRadius: 20,
        paddingVertical: 24,
        alignItems: 'center',
        shadowColor: '#7c6cfa',
        shadowOpacity: 0.5,
        shadowRadius: 15,
        elevation: 8
    },
    btnContent: { flexDirection: 'row', alignItems: 'center' },
    btnIcon: { color: '#fff', fontSize: 18, marginRight: 10 },
    btnText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },

    guideCard: {
        marginTop: 40,
        backgroundColor: '#161622',
        padding: 20,
        borderRadius: 16,
        borderLeftWidth: 4,
        borderLeftColor: '#7c6cfa'
    },
    guideLabel: { color: '#7c6cfa', fontSize: 12, fontWeight: 'bold', marginBottom: 8 },
    guideText: { color: '#e1e1e6', fontSize: 14, lineHeight: 20 }
});