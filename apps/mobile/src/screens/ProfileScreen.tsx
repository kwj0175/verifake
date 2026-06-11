import React, { useState } from 'react';
import { View, Text, SafeAreaView, TouchableOpacity, TextInput, ScrollView, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { PencilIcon, CheckIcon, XMarkIcon, UserIcon, EnvelopeIcon, PhoneIcon } from 'react-native-heroicons/outline';
import { styles } from './ProfileScreen.styles';
import { BottomNavigation } from '../components/BottomNavigaton';

export const ProfileScreen = () => {
    const [isEditing, setIsEditing] = useState(false);
    const [name, setName] = useState('김베리');
    const [email, setEmail] = useState('verifake_user@example.com');
    const [phone, setPhone] = useState('010-1234-5678');

    return (
        <SafeAreaView style={styles.container}>
            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={{ flex: 1 }}
            >
                <ScrollView contentContainerStyle={styles.body} showsVerticalScrollIndicator={false}>
                    {/* 헤더 */}
                    <View style={styles.headerRow}>
                        <View>
                            <Text style={styles.welcomeText}>안녕하세요,</Text>
                            <Text style={styles.headerTitle}>{name}님</Text>
                        </View>

                        {isEditing ? (
                            <View style={styles.actionButtons}>
                                <TouchableOpacity onPress={() => setIsEditing(false)} style={styles.circleBtnCancel}>
                                    <XMarkIcon size={22} color="#ff453a" />
                                </TouchableOpacity>
                                <TouchableOpacity onPress={() => { setIsEditing(false); Alert.alert("성공", "수정 완료!") }} style={styles.circleBtnSave}>
                                    <CheckIcon size={22} color="#32d74b" />
                                </TouchableOpacity>
                            </View>
                        ) : (
                            <TouchableOpacity onPress={() => setIsEditing(true)} style={styles.editFloatingBtn}>
                                <PencilIcon size={18} color="#fff" />
                                <Text style={styles.editFloatingText}>편집</Text>
                            </TouchableOpacity>
                        )}
                    </View>

                    {/* 메인 정보 */}
                    <View style={styles.infoCard}>
                        <Text style={styles.cardHeader}>기본 정보</Text>

                        {/* 이름 */}
                        <View style={styles.fieldItem}>
                            <View style={styles.iconCircle}>
                                <UserIcon size={18} color="#7c6cfa" />
                            </View>
                            <View style={styles.fieldTextContainer}>
                                <Text style={styles.fieldLabel}>이름</Text>
                                {isEditing ? (
                                    <TextInput
                                        style={styles.activeInput}
                                        value={name}
                                        onChangeText={setName}
                                        placeholderTextColor="#444468"
                                    />
                                ) : (
                                    <Text style={styles.fieldValue}>{name}</Text>
                                )}
                            </View>
                        </View>

                        <View style={styles.separator} />

                        {/* 이메일 */}
                        <View style={styles.fieldItem}>
                            <View style={styles.iconCircle}>
                                <EnvelopeIcon size={18} color="#7c6cfa" />
                            </View>
                            <View style={styles.fieldTextContainer}>
                                <Text style={styles.fieldLabel}>이메일</Text>
                                <Text style={styles.readOnlyValue}>{email}</Text>
                            </View>
                        </View>

                        <View style={styles.separator} />

                        {/* 전화번호 */}
                        <View style={styles.fieldItem}>
                            <View style={styles.iconCircle}>
                                <PhoneIcon size={18} color="#7c6cfa" />
                            </View>
                            <View style={styles.fieldTextContainer}>
                                <Text style={styles.fieldLabel}>전화번호</Text>
                                {isEditing ? (
                                    <TextInput
                                        style={styles.activeInput}
                                        value={phone}
                                        onChangeText={setPhone}
                                        keyboardType="phone-pad"
                                    />
                                ) : (
                                    <Text style={styles.fieldValue}>{phone}</Text>
                                )}
                            </View>
                        </View>
                    </View>

                    <View style={styles.accountBadge}>
                        <Text style={styles.badgeText}>Verified Account</Text>
                    </View>
                </ScrollView>
            </KeyboardAvoidingView>

            <BottomNavigation activeRoute="Profile" />
        </SafeAreaView>
    );
};