#!/bin/bash
# Enhanced cross-platform alert sound script for Claude Code hooks
# Usage: ./alert-sound.sh [success|error|info|warning|critical|timeout|permission|validation] [optional-message]

SOUND_TYPE=${1:-success}
MESSAGE=${2:-"Task completed"}

# Function to get appropriate macOS sound
get_macos_sound() {
    case $1 in
        "success")
            echo "/System/Library/Sounds/Glass.aiff"
            ;;
        "error")
            echo "/System/Library/Sounds/Sosumi.aiff"
            ;;
        "critical")
            echo "/System/Library/Sounds/Basso.aiff"
            ;;
        "warning")
            echo "/System/Library/Sounds/Funk.aiff"
            ;;
        "info")
            echo "/System/Library/Sounds/Ping.aiff"
            ;;
        "timeout")
            echo "/System/Library/Sounds/Tink.aiff"
            ;;
        "permission")
            echo "/System/Library/Sounds/Hero.aiff"
            ;;
        "validation")
            echo "/System/Library/Sounds/Blow.aiff"
            ;;
        *)
            echo "/System/Library/Sounds/Ping.aiff"
            ;;
    esac
}

# Function to get appropriate notification icon
get_notification_icon() {
    case $1 in
        "success") echo "✅" ;;
        "error") echo "❌" ;;
        "critical") echo "🚨" ;;
        "warning") echo "⚠️" ;;
        "info") echo "ℹ️" ;;
        "timeout") echo "⏰" ;;
        "permission") echo "🔒" ;;
        "validation") echo "🔍" ;;
        *) echo "📢" ;;
    esac
}

# Detect OS and play appropriate sound
case "$(uname)" in
    "Darwin")
        # macOS
        sound_file=$(get_macos_sound "$SOUND_TYPE")
        afplay "$sound_file" 2>/dev/null || echo -e '\a'
        
        # Enhanced notification with icon
        icon=$(get_notification_icon "$SOUND_TYPE")
        osascript -e "display notification \"$icon $MESSAGE\" with title \"Claude Code\" subtitle \"$SOUND_TYPE\" sound name \"$(basename "$sound_file" .aiff)\"" 2>/dev/null
        ;;
        
    "Linux")
        # Linux with enhanced sound handling
        if command -v paplay &> /dev/null; then
            case $SOUND_TYPE in
                "success")
                    paplay /usr/share/sounds/alsa/Front_Right.wav 2>/dev/null || echo -e '\a'
                    ;;
                "error"|"critical")
                    paplay /usr/share/sounds/alsa/Side_Left.wav 2>/dev/null || echo -e '\a\a'
                    ;;
                "warning")
                    paplay /usr/share/sounds/alsa/Rear_Center.wav 2>/dev/null || echo -e '\a'
                    ;;
                "timeout")
                    # Multiple short beeps for timeout
                    for i in {1..3}; do
                        paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || echo -e '\a'
                        sleep 0.1
                    done
                    ;;
                "permission")
                    paplay /usr/share/sounds/alsa/Side_Right.wav 2>/dev/null || echo -e '\a'
                    ;;
                *)
                    paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || echo -e '\a'
                    ;;
            esac
        else
            # Enhanced fallback beeps
            case $SOUND_TYPE in
                "critical"|"error")
                    echo -e '\a\a'
                    ;;
                "timeout")
                    echo -e '\a' && sleep 0.1 && echo -e '\a' && sleep 0.1 && echo -e '\a'
                    ;;
                *)
                    echo -e '\a'
                    ;;
            esac
        fi
        
        # Enhanced notification with urgency levels
        if command -v notify-send &> /dev/null; then
            icon=$(get_notification_icon "$SOUND_TYPE")
            case $SOUND_TYPE in
                "critical"|"error")
                    notify-send -u critical "Claude Code" "$icon $MESSAGE"
                    ;;
                "warning"|"timeout"|"permission")
                    notify-send -u normal "Claude Code" "$icon $MESSAGE"
                    ;;
                *)
                    notify-send -u low "Claude Code" "$icon $MESSAGE"
                    ;;
            esac
        fi
        ;;
        
    "MINGW"* | "CYGWIN"* | "MSYS"*)
        # Windows (Git Bash/WSL) with enhanced sound patterns
        case $SOUND_TYPE in
            "success")
                powershell.exe -c "[console]::beep(800,300)" 2>/dev/null
                ;;
            "error")
                powershell.exe -c "[console]::beep(400,500)" 2>/dev/null
                ;;
            "critical")
                powershell.exe -c "[console]::beep(300,800)" 2>/dev/null
                ;;
            "warning")
                powershell.exe -c "[console]::beep(600,400)" 2>/dev/null
                ;;
            "timeout")
                powershell.exe -c "for(\$i=0;\$i -lt 3;\$i++){[console]::beep(500,150); Start-Sleep -m 100}" 2>/dev/null
                ;;
            "permission")
                powershell.exe -c "[console]::beep(700,600)" 2>/dev/null
                ;;
            "validation")
                powershell.exe -c "[console]::beep(750,200)" 2>/dev/null
                ;;
            *)
                powershell.exe -c "[console]::beep(600,200)" 2>/dev/null
                ;;
        esac
        ;;
        
    *)
        # Enhanced fallback for unknown systems
        case $SOUND_TYPE in
            "critical"|"error")
                echo -e '\a\a'
                ;;
            "timeout")
                echo -e '\a' && sleep 0.2 && echo -e '\a' && sleep 0.2 && echo -e '\a'
                ;;
            *)
                echo -e '\a'
                ;;
        esac
        ;;
esac

# Enhanced console output with icons and colors
icon=$(get_notification_icon "$SOUND_TYPE")
case $SOUND_TYPE in
    "success")
        echo -e "\033[32m$icon Claude Code: $MESSAGE ($SOUND_TYPE)\033[0m"
        ;;
    "error"|"critical")
        echo -e "\033[31m$icon Claude Code: $MESSAGE ($SOUND_TYPE)\033[0m"
        ;;
    "warning"|"timeout"|"permission")
        echo -e "\033[33m$icon Claude Code: $MESSAGE ($SOUND_TYPE)\033[0m"
        ;;
    *)
        echo -e "\033[36m$icon Claude Code: $MESSAGE ($SOUND_TYPE)\033[0m"
        ;;
esac