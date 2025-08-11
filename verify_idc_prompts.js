// Run this script in the browser console when on the IDC Configuration tab
// It will highlight and verify the system prompt text areas

console.clear();
console.log('🔍 IDC Configuration System Prompts Verification');
console.log('=' .repeat(50));

// Step 1: Check if we're on the IDC Configuration tab
const isOnIDCConfig = window.location.pathname.includes('idc') || 
                      document.querySelector('[aria-controls="tabpanel-5"]')?.getAttribute('aria-selected') === 'true';

if (!isOnIDCConfig) {
    console.warn('⚠️ Please navigate to IDC > Configuration tab first!');
} else {
    console.log('✅ On IDC Configuration page');
}

// Step 2: Look for all text areas
const allTextAreas = document.querySelectorAll('textarea');
console.log(`\n📝 Found ${allTextAreas.length} textarea elements`);

// Step 3: Find system prompt text areas specifically
let extractionPromptField = null;
let validationPromptField = null;

allTextAreas.forEach((textarea, index) => {
    // Check parent elements for labels or nearby text
    const parentElement = textarea.closest('.MuiGrid-item');
    const nearbyText = parentElement?.textContent || '';
    
    if (nearbyText.includes('Extraction System Prompt') || 
        nearbyText.includes('extraction') && nearbyText.includes('prompt')) {
        extractionPromptField = textarea;
        console.log(`\n✅ FOUND Extraction System Prompt field (textarea #${index + 1})`);
        console.log(`   Current value length: ${textarea.value.length} characters`);
        console.log(`   First 100 chars: "${textarea.value.substring(0, 100)}..."`);
    }
    
    if (nearbyText.includes('Validation System Prompt') || 
        nearbyText.includes('validation') && nearbyText.includes('prompt')) {
        validationPromptField = textarea;
        console.log(`\n✅ FOUND Validation System Prompt field (textarea #${index + 1})`);
        console.log(`   Current value length: ${textarea.value.length} characters`);
        console.log(`   First 100 chars: "${textarea.value.substring(0, 100)}..."`);
    }
});

// Step 4: Highlight the fields if found
if (extractionPromptField) {
    // Highlight extraction prompt with blue border
    extractionPromptField.style.border = '3px solid #2196f3';
    extractionPromptField.style.boxShadow = '0 0 10px rgba(33, 150, 243, 0.5)';
    extractionPromptField.scrollIntoView({ behavior: 'smooth', block: 'center' });
    console.log('\n🔵 Extraction prompt field highlighted in BLUE');
} else {
    console.error('\n❌ Extraction System Prompt field NOT FOUND!');
}

if (validationPromptField) {
    // Highlight validation prompt with orange border
    setTimeout(() => {
        validationPromptField.style.border = '3px solid #ff9800';
        validationPromptField.style.boxShadow = '0 0 10px rgba(255, 152, 0, 0.5)';
    }, 1000);
    console.log('🟠 Validation prompt field will be highlighted in ORANGE (1 second delay)');
} else {
    console.error('\n❌ Validation System Prompt field NOT FOUND!');
}

// Step 5: Check if fields might be hidden
if (!extractionPromptField && !validationPromptField && allTextAreas.length === 0) {
    console.error('\n❌ NO text areas found at all! Possible issues:');
    console.log('   1. Component not rendered yet - try refreshing the page');
    console.log('   2. Different tab selected - make sure you\'re on Configuration tab');
    console.log('   3. Loading state - wait for data to load');
    
    // Check for loading indicators
    const loadingIndicator = document.querySelector('.MuiCircularProgress-root');
    if (loadingIndicator) {
        console.log('   ⏳ Page is still loading, please wait...');
    }
}

// Step 6: Provide additional debugging info
console.log('\n📊 Additional Information:');
console.log('   Total MUI Cards:', document.querySelectorAll('.MuiCard-root').length);
console.log('   Total Grid items:', document.querySelectorAll('.MuiGrid-item').length);
console.log('   Total TextFields:', document.querySelectorAll('.MuiTextField-root').length);

// Step 7: Check React DevTools
console.log('\n💡 Tips:');
console.log('   - If fields are not visible, try: Ctrl+Shift+R (hard refresh)');
console.log('   - Clear browser cache: Chrome DevTools > Application > Storage > Clear site data');
console.log('   - Check React DevTools to see if IDCConfigurationPanel is mounted');
console.log('   - Look for any console errors above this message');

// Return summary
const summary = {
    extractionPromptFound: !!extractionPromptField,
    validationPromptFound: !!validationPromptField,
    totalTextAreas: allTextAreas.length,
    isOnCorrectPage: isOnIDCConfig
};

console.log('\n📋 Summary:', summary);
return summary;