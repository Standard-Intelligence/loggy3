/*
```
pip install maturin
maturin develop
```


```python
from loggy3 import chunk_tokenizer

tokens = chunk_tokenizer("/path/to/chunk_directory", "Mac")  # or "Windows"

for seq, token in tokens:
    print(f"Sequence: {seq}, Token: {token}")
```
*/


use std::collections::HashMap;

macro_rules! enum_with_count {
    (enum $name:ident { $($variant:ident),* $(,)? } count=$count_name:ident) => {
        #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
        pub enum $name {
            $($variant),*
        }

        
        impl $name {
            pub fn as_usize(&self) -> usize {
                *self as usize
            }
            
            pub fn from_usize(value: usize) -> Self {
                match value {
                    $(
                        x if x == $name::$variant as usize => $name::$variant,
                    )*
                    _ => panic!("Invalid value for {}: {}", stringify!($name), value),
                }
            }
        }

        
        impl From<u8> for $name {
            fn from(value: u8) -> Self {
                match value {
                    $(
                        x if x == $name::$variant as u8 => $name::$variant,
                    )*
                    _ => panic!("Invalid value for {}: {}", stringify!($name), value),
                }
            }
        }

        
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self)
            }
        }

        
        
        pub const $count_name: usize = 0 $(+ { let _ = $name::$variant; 1 })*;
    };
}

enum_with_count! {
    enum UnifiedKey {
        
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        
        
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        
        
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
        F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
        F21, F22, F23, F24,
        
        
        Return, Tab, Space, Delete, Escape, Backspace,
        
        
        Command, RightCommand, 
        Shift, RightShift, 
        Option, RightOption, Alt, RightAlt,
        Control, RightControl,
        Function, CapsLock, NumLock, ScrollLock,
        
        
        LeftArrow, RightArrow, UpArrow, DownArrow,
        Home, End, PageUp, PageDown,
        Help, ForwardDelete,
        
        
        Keypad0, Keypad1, Keypad2, Keypad3, Keypad4,
        Keypad5, Keypad6, Keypad7, Keypad8, Keypad9,
        KeypadClear, KeypadEnter, KeypadEquals,
        KeypadPlus, KeypadMinus, KeypadMultiply, KeypadDivide, KeypadDecimal,
        
        
        Equal, Minus, LeftBracket, RightBracket,
        Quote, Semicolon, Backslash, Comma, Slash, Period, Grave,
        
        
        VolumeUp, VolumeDown, Mute,
        BrowserBack, BrowserForward, BrowserRefresh, BrowserStop,
        BrowserSearch, BrowserFavorites, BrowserHome,
        MediaNextTrack, MediaPrevTrack, MediaStop, MediaPlayPause,
        
        
        LeftMouse, RightMouse, MiddleMouse, X1Mouse, X2Mouse,
        
        
        LeftWindows, RightWindows, Applications, Sleep,
        
        
        IsoSection, JisYen, JisUnderscore, JisKeypadComma, JisEisu, JisKana,
        
        
        PrintScreen, Insert, Pause, Menu,
        UnknownMac, UnknownWindows
    } count=COUNT_OF_VARIANTS
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Platform {
    Mac,
    Windows,
}

fn init_mac_keymap() -> HashMap<u32, UnifiedKey> {
    let mut map = HashMap::new();
    
    
    map.insert(0x00, UnifiedKey::A);
    map.insert(0x01, UnifiedKey::S);
    map.insert(0x02, UnifiedKey::D);
    map.insert(0x03, UnifiedKey::F);
    map.insert(0x04, UnifiedKey::H);
    map.insert(0x05, UnifiedKey::G);
    map.insert(0x06, UnifiedKey::Z);
    map.insert(0x07, UnifiedKey::X);
    map.insert(0x08, UnifiedKey::C);
    map.insert(0x09, UnifiedKey::V);
    map.insert(0x0B, UnifiedKey::B);
    map.insert(0x0C, UnifiedKey::Q);
    map.insert(0x0D, UnifiedKey::W);
    map.insert(0x0E, UnifiedKey::E);
    map.insert(0x0F, UnifiedKey::R);
    map.insert(0x10, UnifiedKey::Y);
    map.insert(0x11, UnifiedKey::T);
    map.insert(0x1F, UnifiedKey::O);
    map.insert(0x20, UnifiedKey::U);
    map.insert(0x22, UnifiedKey::I);
    map.insert(0x23, UnifiedKey::P);
    map.insert(0x25, UnifiedKey::L);
    map.insert(0x26, UnifiedKey::J);
    map.insert(0x28, UnifiedKey::K);
    map.insert(0x2D, UnifiedKey::N);
    map.insert(0x2E, UnifiedKey::M);
    
    
    map.insert(0x12, UnifiedKey::Num1);
    map.insert(0x13, UnifiedKey::Num2);
    map.insert(0x14, UnifiedKey::Num3);
    map.insert(0x15, UnifiedKey::Num4);
    map.insert(0x17, UnifiedKey::Num5);
    map.insert(0x16, UnifiedKey::Num6);
    map.insert(0x1A, UnifiedKey::Num7);
    map.insert(0x1C, UnifiedKey::Num8);
    map.insert(0x19, UnifiedKey::Num9);
    map.insert(0x1D, UnifiedKey::Num0);
    
    
    map.insert(0x18, UnifiedKey::Equal);
    map.insert(0x1B, UnifiedKey::Minus);
    map.insert(0x21, UnifiedKey::LeftBracket);
    map.insert(0x1E, UnifiedKey::RightBracket);
    map.insert(0x27, UnifiedKey::Quote);
    map.insert(0x29, UnifiedKey::Semicolon);
    map.insert(0x2A, UnifiedKey::Backslash);
    map.insert(0x2B, UnifiedKey::Comma);
    map.insert(0x2C, UnifiedKey::Slash);
    map.insert(0x2F, UnifiedKey::Period);
    map.insert(0x32, UnifiedKey::Grave);
    
    
    map.insert(0x7A, UnifiedKey::F1);
    map.insert(0x78, UnifiedKey::F2);
    map.insert(0x63, UnifiedKey::F3);
    map.insert(0x76, UnifiedKey::F4);
    map.insert(0x60, UnifiedKey::F5);
    map.insert(0x61, UnifiedKey::F6);
    map.insert(0x62, UnifiedKey::F7);
    map.insert(0x64, UnifiedKey::F8);
    map.insert(0x65, UnifiedKey::F9);
    map.insert(0x6D, UnifiedKey::F10);
    map.insert(0x67, UnifiedKey::F11);
    map.insert(0x6F, UnifiedKey::F12);
    map.insert(0x69, UnifiedKey::F13);
    map.insert(0x6B, UnifiedKey::F14);
    map.insert(0x71, UnifiedKey::F15);
    map.insert(0x6A, UnifiedKey::F16);
    map.insert(0x40, UnifiedKey::F17);
    map.insert(0x4F, UnifiedKey::F18);
    map.insert(0x50, UnifiedKey::F19);
    map.insert(0x5A, UnifiedKey::F20);
    
    
    map.insert(0x24, UnifiedKey::Return);
    map.insert(0x30, UnifiedKey::Tab);
    map.insert(0x31, UnifiedKey::Space);
    map.insert(0x33, UnifiedKey::Delete);
    map.insert(0x35, UnifiedKey::Escape);
    map.insert(0x37, UnifiedKey::Command);
    map.insert(0x38, UnifiedKey::Shift);
    map.insert(0x39, UnifiedKey::CapsLock);
    map.insert(0x3A, UnifiedKey::Option);
    map.insert(0x3B, UnifiedKey::Control);
    map.insert(0x36, UnifiedKey::RightCommand);
    map.insert(0x3C, UnifiedKey::RightShift);
    map.insert(0x3D, UnifiedKey::RightOption);
    map.insert(0x3E, UnifiedKey::RightControl);
    map.insert(0x3F, UnifiedKey::Function);
    map.insert(0x47, UnifiedKey::KeypadClear);
    map.insert(0x4C, UnifiedKey::KeypadEnter);
    
    
    map.insert(0x7B, UnifiedKey::LeftArrow);
    map.insert(0x7C, UnifiedKey::RightArrow);
    map.insert(0x7D, UnifiedKey::DownArrow);
    map.insert(0x7E, UnifiedKey::UpArrow);
    
    
    map.insert(0x72, UnifiedKey::Help);
    map.insert(0x73, UnifiedKey::Home);
    map.insert(0x74, UnifiedKey::PageUp);
    map.insert(0x75, UnifiedKey::ForwardDelete);
    map.insert(0x77, UnifiedKey::End);
    map.insert(0x79, UnifiedKey::PageDown);
    
    
    map.insert(0x52, UnifiedKey::Keypad0);
    map.insert(0x53, UnifiedKey::Keypad1);
    map.insert(0x54, UnifiedKey::Keypad2);
    map.insert(0x55, UnifiedKey::Keypad3);
    map.insert(0x56, UnifiedKey::Keypad4);
    map.insert(0x57, UnifiedKey::Keypad5);
    map.insert(0x58, UnifiedKey::Keypad6);
    map.insert(0x59, UnifiedKey::Keypad7);
    map.insert(0x5B, UnifiedKey::Keypad8);
    map.insert(0x5C, UnifiedKey::Keypad9);
    map.insert(0x41, UnifiedKey::KeypadDecimal);
    map.insert(0x43, UnifiedKey::KeypadMultiply);
    map.insert(0x45, UnifiedKey::KeypadPlus);
    map.insert(0x4B, UnifiedKey::KeypadDivide);
    map.insert(0x4E, UnifiedKey::KeypadMinus);
    map.insert(0x51, UnifiedKey::KeypadEquals);
    
    
    map.insert(0x48, UnifiedKey::VolumeUp);
    map.insert(0x49, UnifiedKey::VolumeDown);
    map.insert(0x4A, UnifiedKey::Mute);
    
    
    map.insert(0x0A, UnifiedKey::IsoSection);
    map.insert(0x5D, UnifiedKey::JisYen);
    map.insert(0x5E, UnifiedKey::JisUnderscore);
    map.insert(0x5F, UnifiedKey::JisKeypadComma);
    map.insert(0x66, UnifiedKey::JisEisu);
    map.insert(0x68, UnifiedKey::JisKana);
    
    map
}

fn init_mac_modifier_map() -> HashMap<&'static str, UnifiedKey> {
    let mut map = HashMap::new();
    
    map.insert("shift", UnifiedKey::Shift);
    map.insert("control", UnifiedKey::Control);
    map.insert("alternate", UnifiedKey::Option);
    map.insert("command", UnifiedKey::Command);

    map
}

static MAC_IGNORED_MODIFIERS: [&str; 5] = ["alphaShift", "help", "secondaryFn", "numericPad", "nonCoalesced"];
    
    
fn init_windows_keymap() -> HashMap<u32, UnifiedKey> {
    let mut map = HashMap::new();
    
    
    map.insert(0x01, UnifiedKey::LeftMouse);
    map.insert(0x02, UnifiedKey::RightMouse);
    map.insert(0x04, UnifiedKey::MiddleMouse);
    map.insert(0x05, UnifiedKey::X1Mouse);
    map.insert(0x06, UnifiedKey::X2Mouse);
    
    
    map.insert(0x08, UnifiedKey::Backspace);
    map.insert(0x09, UnifiedKey::Tab);
    map.insert(0x0C, UnifiedKey::KeypadClear);
    map.insert(0x0D, UnifiedKey::Return);
    map.insert(0x10, UnifiedKey::Shift);
    map.insert(0x11, UnifiedKey::Control);
    map.insert(0x12, UnifiedKey::Alt);
    map.insert(0x13, UnifiedKey::Pause);
    map.insert(0x14, UnifiedKey::CapsLock);
    map.insert(0x1B, UnifiedKey::Escape);
    map.insert(0x20, UnifiedKey::Space);
    map.insert(0x21, UnifiedKey::PageUp);
    map.insert(0x22, UnifiedKey::PageDown);
    map.insert(0x23, UnifiedKey::End);
    map.insert(0x24, UnifiedKey::Home);
    
    
    map.insert(0x25, UnifiedKey::LeftArrow);
    map.insert(0x26, UnifiedKey::UpArrow);
    map.insert(0x27, UnifiedKey::RightArrow);
    map.insert(0x28, UnifiedKey::DownArrow);
    
    
    map.insert(0x2C, UnifiedKey::PrintScreen);
    map.insert(0x2D, UnifiedKey::Insert);
    map.insert(0x2E, UnifiedKey::Delete);
    map.insert(0x2F, UnifiedKey::Help);
    
    
    for i in 0..10 {
        map.insert(0x30 + i, match i {
            0 => UnifiedKey::Num0,
            1 => UnifiedKey::Num1,
            2 => UnifiedKey::Num2,
            3 => UnifiedKey::Num3,
            4 => UnifiedKey::Num4,
            5 => UnifiedKey::Num5,
            6 => UnifiedKey::Num6,
            7 => UnifiedKey::Num7,
            8 => UnifiedKey::Num8,
            9 => UnifiedKey::Num9,
            _ => unreachable!(),
        });
    }
    
    
    for i in 0..26 {
        map.insert(0x41 + i, match i {
            0 => UnifiedKey::A,
            1 => UnifiedKey::B,
            2 => UnifiedKey::C,
            3 => UnifiedKey::D,
            4 => UnifiedKey::E,
            5 => UnifiedKey::F,
            6 => UnifiedKey::G,
            7 => UnifiedKey::H,
            8 => UnifiedKey::I,
            9 => UnifiedKey::J,
            10 => UnifiedKey::K,
            11 => UnifiedKey::L,
            12 => UnifiedKey::M,
            13 => UnifiedKey::N,
            14 => UnifiedKey::O,
            15 => UnifiedKey::P,
            16 => UnifiedKey::Q,
            17 => UnifiedKey::R,
            18 => UnifiedKey::S,
            19 => UnifiedKey::T,
            20 => UnifiedKey::U,
            21 => UnifiedKey::V,
            22 => UnifiedKey::W,
            23 => UnifiedKey::X,
            24 => UnifiedKey::Y,
            25 => UnifiedKey::Z,
            _ => unreachable!(),
        });
    }
    
    
    map.insert(0x5B, UnifiedKey::LeftWindows);
    map.insert(0x5C, UnifiedKey::RightWindows);
    map.insert(0x5D, UnifiedKey::Applications);
    map.insert(0x5F, UnifiedKey::Sleep);
    
    
    map.insert(0x60, UnifiedKey::Keypad0);
    map.insert(0x61, UnifiedKey::Keypad1);
    map.insert(0x62, UnifiedKey::Keypad2);
    map.insert(0x63, UnifiedKey::Keypad3);
    map.insert(0x64, UnifiedKey::Keypad4);
    map.insert(0x65, UnifiedKey::Keypad5);
    map.insert(0x66, UnifiedKey::Keypad6);
    map.insert(0x67, UnifiedKey::Keypad7);
    map.insert(0x68, UnifiedKey::Keypad8);
    map.insert(0x69, UnifiedKey::Keypad9);
    map.insert(0x6A, UnifiedKey::KeypadMultiply);
    map.insert(0x6B, UnifiedKey::KeypadPlus);
    map.insert(0x6D, UnifiedKey::KeypadMinus);
    map.insert(0x6E, UnifiedKey::KeypadDecimal);
    map.insert(0x6F, UnifiedKey::KeypadDivide);
    
    
    for i in 0..24 {
        if i < 20 {  
            map.insert(0x70 + i, match i {
                0 => UnifiedKey::F1,
                1 => UnifiedKey::F2,
                2 => UnifiedKey::F3,
                3 => UnifiedKey::F4,
                4 => UnifiedKey::F5,
                5 => UnifiedKey::F6,
                6 => UnifiedKey::F7,
                7 => UnifiedKey::F8,
                8 => UnifiedKey::F9,
                9 => UnifiedKey::F10,
                10 => UnifiedKey::F11,
                11 => UnifiedKey::F12,
                12 => UnifiedKey::F13,
                13 => UnifiedKey::F14,
                14 => UnifiedKey::F15,
                15 => UnifiedKey::F16,
                16 => UnifiedKey::F17,
                17 => UnifiedKey::F18,
                18 => UnifiedKey::F19,
                19 => UnifiedKey::F20,
                _ => unreachable!(),
            });
        } else {  
            map.insert(0x70 + i, match i {
                20 => UnifiedKey::F21,
                21 => UnifiedKey::F22,
                22 => UnifiedKey::F23,
                23 => UnifiedKey::F24,
                _ => unreachable!(),
            });
        }
    }
    
    
    map.insert(0x90, UnifiedKey::NumLock);
    map.insert(0x91, UnifiedKey::ScrollLock);
    map.insert(0xA0, UnifiedKey::Shift);
    map.insert(0xA1, UnifiedKey::RightShift);
    map.insert(0xA2, UnifiedKey::Control);
    map.insert(0xA3, UnifiedKey::RightControl);
    map.insert(0xA4, UnifiedKey::Alt);
    map.insert(0xA5, UnifiedKey::RightAlt);
    
    
    map.insert(0xA6, UnifiedKey::BrowserBack);
    map.insert(0xA7, UnifiedKey::BrowserForward);
    map.insert(0xA8, UnifiedKey::BrowserRefresh);
    map.insert(0xA9, UnifiedKey::BrowserStop);
    map.insert(0xAA, UnifiedKey::BrowserSearch);
    map.insert(0xAB, UnifiedKey::BrowserFavorites);
    map.insert(0xAC, UnifiedKey::BrowserHome);
    
    
    map.insert(0xAD, UnifiedKey::Mute);
    map.insert(0xAE, UnifiedKey::VolumeDown);
    map.insert(0xAF, UnifiedKey::VolumeUp);
    map.insert(0xB0, UnifiedKey::MediaNextTrack);
    map.insert(0xB1, UnifiedKey::MediaPrevTrack);
    map.insert(0xB2, UnifiedKey::MediaStop);
    map.insert(0xB3, UnifiedKey::MediaPlayPause);
    
    
    map.insert(0xBA, UnifiedKey::Semicolon);
    map.insert(0xBB, UnifiedKey::Equal);
    map.insert(0xBC, UnifiedKey::Comma);
    map.insert(0xBD, UnifiedKey::Minus);
    map.insert(0xBE, UnifiedKey::Period);
    map.insert(0xBF, UnifiedKey::Slash);
    map.insert(0xC0, UnifiedKey::Grave);
    map.insert(0xDB, UnifiedKey::LeftBracket);
    map.insert(0xDC, UnifiedKey::Backslash);
    map.insert(0xDD, UnifiedKey::RightBracket);
    map.insert(0xDE, UnifiedKey::Quote);
    
    map
}

lazy_static::lazy_static! {
    static ref MAC_KEYMAP: HashMap<u32, UnifiedKey> = init_mac_keymap();
    static ref MAC_MODIFIER_MAP: HashMap<&'static str, UnifiedKey> = init_mac_modifier_map();
    static ref WINDOWS_KEYMAP: HashMap<u32, UnifiedKey> = init_windows_keymap();
}


fn get_unified_key(platform: Platform, keycode: u32) -> UnifiedKey {
    match platform {
        Platform::Mac => MAC_KEYMAP
            .get(&keycode)
            .cloned()
            .unwrap_or(UnifiedKey::UnknownMac),
        
        Platform::Windows => WINDOWS_KEYMAP
            .get(&keycode)
            .cloned()
            .unwrap_or(UnifiedKey::UnknownWindows),
    }
}

const KEY_COUNT: usize = COUNT_OF_VARIANTS * 2;
const WHEEL_START: usize = KEY_COUNT + 81;
const FRAME_TOKEN_DISPLAY_1: usize = WHEEL_START + 81;
const FRAME_TOKEN_DISPLAY_2: usize = WHEEL_START + 82;
const FRAME_TOKEN_DISPLAY_3: usize = WHEEL_START + 83;
const FRAME_TOKEN_DISPLAY_4: usize = WHEEL_START + 84;


fn bin_coordinate(coord: i32) -> usize {
    if coord < -3 {
        0
    } else if coord > 3 {
        8
    } else {
        (coord + 3) as usize + 1
    }
}

fn bin_coordinates(x: i32, y: i32, is_wheel: bool) -> usize {
    let x_bin = bin_coordinate(x);
    let y_bin = bin_coordinate(y);
    let base_token = y_bin * 9 + x_bin + KEY_COUNT;
    if is_wheel {
        base_token + COUNT_OF_VARIANTS
    } else {
        base_token
    }
}

fn unbin_coordinate(bin: usize) -> i32 {
    if bin == 0 {
        -4
    } else if bin == 8 {
        4
    } else {
        (bin as i32) - 4
    }
}

fn unbin_value(value: usize) -> (i32, i32, bool) {
    let is_wheel = value >= WHEEL_START;
    let value = if is_wheel {
        value - WHEEL_START
    } else {
        value - KEY_COUNT
    };
    let y_bin = value / 9;
    let x_bin = value % 9;
    (unbin_coordinate(x_bin), unbin_coordinate(y_bin), is_wheel)
}


fn unified_to_keytoken(unified_key: UnifiedKey, pressed: bool) -> usize {
    let base_code = unified_key.as_usize();
    if pressed {
        base_code + COUNT_OF_VARIANTS
    } else {
        base_code
    }
}

fn keytoken_to_unified(keytoken: usize) -> (UnifiedKey, bool) {
    if keytoken >= COUNT_OF_VARIANTS {
        (UnifiedKey::from_usize(keytoken - COUNT_OF_VARIANTS), true)
    } else {
        (UnifiedKey::from_usize(keytoken), false)
    }
}

pub fn token_to_readable(token: usize) -> String {
    if token == FRAME_TOKEN_DISPLAY_1 {
        format!("🎥 FRAME (Display 1)")
    } else if token == FRAME_TOKEN_DISPLAY_2 {
        format!("🎥 FRAME (Display 2)")
    } else if token == FRAME_TOKEN_DISPLAY_3 {
        format!("🎥 FRAME (Display 3)")
    } else if token == FRAME_TOKEN_DISPLAY_4 {
        format!("🎥 FRAME (Display 4)")
    } else if token < KEY_COUNT {
        let (unified_key, pressed) = keytoken_to_unified(token);
        format!("🔑 {} {}", unified_key, if pressed { "Pressed" } else { "Released" })
    } else {
        let (x, y, is_wheel) = unbin_value(token);
        format!("{} ({}, {})", if is_wheel { "📜" } else { "🐭" }, x, y)
    }
}

pub fn print_token_sequence(tokens: &[(usize, usize)]) {
    for (seq, token) in tokens {
        println!("[{}] {}", seq, token_to_readable(*token));
    }
}

fn extract_sequence_number(line: &str) -> usize {
    if let Some(start_idx) = line.find("[(") {
        if let Some(end_idx) = line[start_idx + 2..].find(",") {
            let seq_str = &line[start_idx + 2..start_idx + 2 + end_idx];
            seq_str.parse::<usize>().unwrap_or(0)
        } else {
            0
        }
    } else {
        0
    }
}



fn parse_mac_keylog(keylog: &str) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let lines: Vec<&str> = keylog.lines().collect();
    
    for line in lines {
        let event_data = line.trim();
        if event_data.is_empty() {
            continue;
        }

        let seq_num = extract_sequence_number(event_data);

        let type_string = "\"type\": ";
        let keycode_string = "\"keycode\": ";
        let flagschanged_string = "\"flagsChanged\": {";
        if let Some(type_start) = event_data.find(type_string) {
            let type_value_start = type_start + type_string.len();
            if let Some(type_end) = event_data[type_value_start..].find(",") {
                let event_type = &event_data[type_value_start+1..type_value_start + type_end-1];
                
                match event_type {
                    "key_down" | "key_up" => {
                        if let Some(keycode_start) = event_data.find(keycode_string) {
                            let keycode_value_start = keycode_start + keycode_string.len();
                            if let Some(keycode_end) = event_data[keycode_value_start..].find(",") {
                                if let Ok(keycode) = event_data[keycode_value_start..keycode_value_start + keycode_end].trim_matches('"').parse::<u32>() {
                                    let unified_key = get_unified_key(Platform::Mac, keycode);
                                    let unified_key_num = unified_to_keytoken(unified_key, event_type == "key_down");
                                    tokens.push((seq_num, unified_key_num));
                                } else {
                                    eprintln!("Failed to parse keycode");
                                    eprintln!("Event data: {}", event_data);
                                }
                            }
                        }
                    }
                    "flags_changed" => {
                        if let Some(flagschanged_start) = event_data.find(flagschanged_string) {
                            let flagschanged_value_start = flagschanged_start + flagschanged_string.len();
                            if let Some(flagschanged_end) = event_data[flagschanged_value_start..].find("}") {
                                let flagschanged = &event_data[flagschanged_value_start..flagschanged_value_start + flagschanged_end];

                                for modifier_pair in flagschanged.split(',').map(|s| s.trim()) {
                                    if modifier_pair.is_empty() {
                                        continue;
                                    }
                                    
                                    if let Some(colon_pos) = modifier_pair.find(':') {
                                        let modifier_name = &modifier_pair[..colon_pos].trim_matches('"');
                                        let state_name = &modifier_pair[colon_pos+1..].trim().trim_matches('"');
                                        let state_value = match *state_name {
                                            "pressed" => "⬇️",
                                            "released" => "⬆️",
                                            _ => unreachable!(),
                                        };

                                        
                                        if MAC_IGNORED_MODIFIERS.contains(&modifier_name) {
                                            continue;
                                        }
                                        
                                        if let Some(unified_key) = MAC_MODIFIER_MAP.get(modifier_name) {
                                            let unified_key_num = unified_to_keytoken(*unified_key, state_value == "⬇️");
                                            tokens.push((seq_num, unified_key_num));
                                        } else {
                                            eprintln!("Unknown modifier: {} ({})", modifier_name, state_value);
                                            eprintln!("Flags changed: {}", flagschanged);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        eprintln!("Unknown event type: {}", event_type);
                    }
                }
            }
        }
    }
    tokens
}


fn parse_mac_mouselog(mouselog: &str) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let lines: Vec<&str> = mouselog.lines().collect();
    
    for line in lines {
        let event_data = line.trim();
        if event_data.is_empty() {
            continue;
        }

        let seq_num = extract_sequence_number(event_data);

        let type_string = "\"type\": ";
        let event_kind_string = "\"eventType\": ";
        let delta_x_string = "\"deltaX\": ";
        let delta_y_string = "\"deltaY\": ";
        let wheel_delta_x_string = "\"pointDeltaAxis1\": ";
        let wheel_delta_y_string = "\"pointDeltaAxis2\": ";

        if let Some(type_start) = event_data.find(type_string) {
            let type_value_start = type_start + type_string.len();
            if let Some(type_end) = event_data[type_value_start..].find(",") {
                let event_type = &event_data[type_value_start+1..type_value_start + type_end-1];

                match event_type {
                    "mouse_movement" => {
                        let dx_start = event_data.find(delta_x_string).unwrap();
                        let dx_value_start = dx_start + delta_x_string.len();
                        let dx_value_end = event_data[dx_value_start..].find(",").unwrap();
                        let dx = &event_data[dx_value_start..dx_value_start + dx_value_end];

                        let dy_start = event_data.find(delta_y_string).unwrap();
                        let dy_value_start = dy_start + delta_y_string.len();
                        let dy_value_end = event_data[dy_value_start..].find(",").unwrap();
                        let dy = &event_data[dy_value_start..dy_value_start + dy_value_end];

                        if let (Ok(dx), Ok(dy)) = (dx.parse::<i32>(), dy.parse::<i32>()) {
                            let bin = bin_coordinates(dx, dy, false);
                            tokens.push((seq_num, bin));
                        } else {
                            eprintln!("Failed to parse delta values: {} {}", dx, dy);
                        }
                    }
                    "scroll_wheel" => {
                        let wheel_delta_x_start = event_data.find(wheel_delta_x_string).unwrap();
                        let wheel_delta_x_value_start = wheel_delta_x_start + wheel_delta_x_string.len();
                        let wheel_delta_x_value_end = event_data[wheel_delta_x_value_start..].find(",").unwrap();
                        let wheel_delta_x = &event_data[wheel_delta_x_value_start..wheel_delta_x_value_start + wheel_delta_x_value_end];

                        let wheel_delta_y_start = event_data.find(wheel_delta_y_string).unwrap();
                        let wheel_delta_y_value_start = wheel_delta_y_start + wheel_delta_y_string.len();
                        let wheel_delta_y_value_end = event_data[wheel_delta_y_value_start..].find("}").unwrap();
                        let wheel_delta_y = &event_data[wheel_delta_y_value_start..wheel_delta_y_value_start + wheel_delta_y_value_end];

                        if let (Ok(wheel_delta_x), Ok(wheel_delta_y)) = (wheel_delta_x.parse::<i32>(), wheel_delta_y.parse::<i32>()) {
                            let bin = bin_coordinates(wheel_delta_x, wheel_delta_y, true);
                            tokens.push((seq_num, bin));
                        } else {
                            eprintln!("Failed to parse wheel delta values: {} {}", wheel_delta_x, wheel_delta_y);
                        }
                    }
                    "mouse_down" | "mouse_up" => {
                        let state_value_start = event_data.find(event_kind_string).unwrap();
                        let state_value_start = state_value_start + event_kind_string.len();
                        let state_value_end = event_data[state_value_start..].find(",").unwrap();
                        let event_kind = &event_data[state_value_start+1..state_value_start + state_value_end-1];

                        let (unified_key, is_pressed) = match event_kind {
                            "LeftMouseDown" => (UnifiedKey::LeftMouse, true),
                            "RightMouseDown" => (UnifiedKey::RightMouse, true),
                            "LeftMouseUp" => (UnifiedKey::LeftMouse, false),
                            "RightMouseUp" => (UnifiedKey::RightMouse, false),
                            _ => unreachable!(),
                        };
                        let unified_key_num = unified_to_keytoken(unified_key, is_pressed);
                        tokens.push((seq_num, unified_key_num));
                    }
                    
                    _ => {
                        eprintln!("Unknown event type: {}", event_type);
                    }
                }
            }
        }
    }
    tokens
}


fn parse_windows_keylog(keylog: &str) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let lines: Vec<&str> = keylog.lines().collect();

    for line in lines {
        let event_data = line.trim();
        if event_data.is_empty() {
            continue;
        }

        let seq_num = extract_sequence_number(event_data);

        if let Some(start_idx) = event_data.find("'") {
            if let Some(end_idx) = event_data.rfind("'") {
                if start_idx < end_idx {
                    let content = &event_data[start_idx + 1..end_idx];
                    let parts: Vec<&str> = content.split(", ").collect();
                    if parts.len() == 2 {
                        let action = parts[0].trim_matches('\'');
                        let key = parts[1].trim_matches('\'');
                        let unified_key = get_unified_key(Platform::Windows, key[3..].parse::<u32>().unwrap());
                        let unified_key_num = unified_to_keytoken(unified_key, action == "press");
                        tokens.push((seq_num, unified_key_num));
                    } else {
                        eprintln!("Invalid format: {}", content);
                    }
                }
            }
        }
    }
    tokens
}


fn parse_windows_mouselog(mouselog: &str) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let lines: Vec<&str> = mouselog.lines().collect();
    
    for line in lines {
        let event_data = line.trim();
        if event_data.is_empty() {
            continue;
        }

        let seq_num = extract_sequence_number(event_data);

        let type_string = "\"type\":\"";
        let delta_x_string = "\"deltaX\":";
        let delta_y_string = "\"deltaY\":";
        let action_string = "\"action\":\"";
        let button_string = "\"button\":\"";

        if let Some(type_start) = event_data.find(type_string) {
            let type_value_start = type_start + type_string.len();
            if let Some(type_end) = event_data[type_value_start..].find(",") {
                let event_type = &event_data[type_value_start..type_value_start + type_end-1];
                match event_type {
                    "Delta" | "Wheel" => {
                        let delta_x_start = event_data.find(delta_x_string).unwrap();
                        let delta_x_value_start = delta_x_start + delta_x_string.len();
                        let delta_x_value_end = event_data[delta_x_value_start..].find(",").unwrap();
                        let delta_x = &event_data[delta_x_value_start..delta_x_value_start + delta_x_value_end];

                        let delta_y_start = event_data.find(delta_y_string).unwrap();
                        let delta_y_value_start = delta_y_start + delta_y_string.len();
                        let delta_y_value_end = event_data[delta_y_value_start..].find(",").unwrap();
                        let delta_y = &event_data[delta_y_value_start..delta_y_value_start + delta_y_value_end];

                        let delta_token = bin_coordinates(
                            delta_x.parse::<i32>().unwrap(), 
                            delta_y.parse::<i32>().unwrap(), 
                            event_type == "Wheel");
                        tokens.push((seq_num, delta_token));
                    }
                    "Button" => {
                        let action_start = event_data.find(action_string).unwrap();
                        let action_value_start = action_start + action_string.len();
                        let action_value_end = event_data[action_value_start..].find(",").unwrap();
                        let action_value = &event_data[action_value_start..action_value_start + action_value_end-1];

                        let button_start = event_data.find(button_string).unwrap();
                        let button_value_start = button_start + button_string.len();
                        let button_value_end = event_data[button_value_start..].find(",").unwrap();
                        let button = &event_data[button_value_start..button_value_start + button_value_end-1];

                        let unified_key = match button {
                            "Left" => UnifiedKey::LeftMouse,
                            "Right" => UnifiedKey::RightMouse,
                            _ => unreachable!(),
                        };
                        let unified_key_num = unified_to_keytoken(unified_key, action_value == "press");
                        tokens.push((seq_num, unified_key_num));
                    }

                    _ => unreachable!(),
                }
            }
        }
    }
    tokens
}

fn parse_frames_log(frameslog: &str, display_num: usize) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let lines: Vec<&str> = frameslog.lines().collect();
    
    let frame_token = match display_num {
        1 => FRAME_TOKEN_DISPLAY_1,
        2 => FRAME_TOKEN_DISPLAY_2,
        3 => FRAME_TOKEN_DISPLAY_3,
        4 => FRAME_TOKEN_DISPLAY_4,
        _ => FRAME_TOKEN_DISPLAY_1, 
    };
    
    for line in lines {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            if let Ok(seq_num) = parts[0].parse::<usize>() {
                tokens.push((seq_num, frame_token));
            }
        }
    }
    
    tokens
}

pub fn chunk_tokenizer(chunk_path: &std::path::Path, os_type: &str) -> Vec<(usize, usize)> {
    let mut chunk_tokens = Vec::new();
    
    
    let keypresses_path = chunk_path.join("keypresses.log");
    if keypresses_path.exists() {
        match std::fs::read_to_string(&keypresses_path) {
            Ok(keylog) => {
                let key_tokens = if os_type == "Windows" {
                    parse_windows_keylog(&keylog)
                } else {
                    parse_mac_keylog(&keylog)
                };
                chunk_tokens.extend(key_tokens);
            },
            Err(e) => {
                eprintln!("Error reading file {}: {}", keypresses_path.display(), e);
            }
        }
    } else {
        println!("No keypresses.log found in {}", chunk_path.display());
    }
    
    
    let mouselog_path = chunk_path.join("mouse.log");
    if mouselog_path.exists() {
        match std::fs::read_to_string(&mouselog_path) {
            Ok(mouselog) => {
                let mouse_tokens = if os_type == "Windows" {
                    parse_windows_mouselog(&mouselog)
                } else {
                    parse_mac_mouselog(&mouselog)
                };
                chunk_tokens.extend(mouse_tokens);
            }
            Err(e) => {
                eprintln!("Error reading file {}: {}", mouselog_path.display(), e);
            }
        }
    } else {
        println!("No mouse.log found in {}", chunk_path.display());
    }
    
    
    let mut display_dirs = Vec::new();
    if let Ok(chunk_entries) = std::fs::read_dir(chunk_path) {
        for chunk_entry in chunk_entries {
            if let Ok(chunk_entry) = chunk_entry {
                let display_path = chunk_entry.path();
                if display_path.is_dir() && display_path.file_name().unwrap_or_default().to_string_lossy().starts_with("display_") {
                    display_dirs.push(display_path);
                }
            }
        }
    }
    
    
    display_dirs.sort_by(|a, b| a.file_name().unwrap_or_default().cmp(&b.file_name().unwrap_or_default()));
    
    
    if display_dirs.is_empty() {
        println!("No display directories found in chunk {}", chunk_path.display());
    } else {
        for (i, display_path) in display_dirs.iter().take(4).enumerate() {
            let display_num = i + 1; 
            let frames_path = display_path.join("frames.log");
            if frames_path.exists() {
                match std::fs::read_to_string(&frames_path) {
                    Ok(frameslog) => {
                        let frame_tokens = parse_frames_log(&frameslog, display_num);
                        chunk_tokens.extend(frame_tokens);
                    }
                    Err(e) => {
                        eprintln!("Error reading file {}: {}", frames_path.display(), e);
                    }
                }
            } else {
                println!("No frames.log found in display directory {}", display_path.display());
            }
        }
    }
    
    
    chunk_tokens.sort_by_key(|(seq, _)| *seq);
    
    chunk_tokens
}

#[allow(dead_code)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <session_directory>", args[0]);
        std::process::exit(1);
    }
    
    let session_dir = &args[1];
    let session_path = std::path::Path::new(session_dir);
    
    if !session_path.exists() || !session_path.is_dir() {
        eprintln!("Error: The specified path '{}' does not exist or is not a directory", session_dir);
        std::process::exit(1);
    };
    
    
    let metadata_path = session_path.join("session_metadata.json");
    let os_type = if metadata_path.exists() {
        match std::fs::read_to_string(&metadata_path) {
            Ok(metadata) => {
                if metadata.contains("\"os_name\": \"Windows\"") {
                    println!("Detected Windows OS");
                    "Windows"
                } else if metadata.contains("\"os_name\": \"Darwin\"") {
                    println!("Detected macOS");
                    "Darwin"
                } else {
                    eprintln!("Unknown OS type in metadata file");
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Error reading metadata file: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("No metadata file found at {}", metadata_path.display());
        std::process::exit(1);
    };
    
    let start_time = std::time::Instant::now();
    
    match std::fs::read_dir(session_path) {
        Ok(entries) => {
            let mut all_tokens = Vec::new();
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    
                    if path.is_dir() && path.file_name().unwrap_or_default().to_string_lossy().starts_with("chunk_") {
                        let chunk_tokens = chunk_tokenizer(&path, os_type);
                        all_tokens.extend(chunk_tokens);
                    }
                }
            }
            
            let elapsed = start_time.elapsed();
            let tokens_per_second = all_tokens.len() as f64 / elapsed.as_secs_f64();
            println!("Token sequence generation completed in {:.2?} ({:.2} tokens/sec)", elapsed, tokens_per_second);
            
            println!("Combined sorted tokens:");
            all_tokens.sort_by_key(|(seq, _)| *seq);
            print_token_sequence(&all_tokens[..all_tokens.len().min(500)]);
        },
        Err(e) => {
            eprintln!("Error reading directory {}: {}", session_dir, e);
            std::process::exit(1);
        }
    }
}
