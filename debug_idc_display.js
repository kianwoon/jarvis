// Debug script to check if IDC Configuration text areas are present but hidden
// Run this in browser console when on the IDC Configuration tab

function debugIDCConfigDisplay() {
    console.log('=== IDC Configuration Debug ===');
    
    // Check for text areas by label
    const allTextFields = document.querySelectorAll('textarea, input[type="text"]');
    console.log(`Total text fields found: ${allTextFields.length}`);
    
    // Look specifically for system prompt fields
    allTextFields.forEach(field => {
        const label = field.parentElement?.querySelector('label')?.textContent || 
                     field.getAttribute('aria-label') || 
                     field.getAttribute('placeholder') || 
                     'No label';
        
        if (label.toLowerCase().includes('prompt') || label.toLowerCase().includes('system')) {
            console.log('\n📝 Found prompt field:');
            console.log('  Label:', label);
            console.log('  Value length:', field.value?.length || 0);
            console.log('  Is visible:', field.offsetParent !== null);
            console.log('  Display style:', window.getComputedStyle(field).display);
            console.log('  Visibility:', window.getComputedStyle(field).visibility);
            console.log('  Opacity:', window.getComputedStyle(field).opacity);
            console.log('  Height:', field.offsetHeight);
            console.log('  Parent element:', field.parentElement);
            
            // Check if it's in a collapsed section
            let parent = field.parentElement;
            while (parent) {
                if (parent.style.display === 'none' || 
                    window.getComputedStyle(parent).display === 'none') {
                    console.log('  ⚠️ Hidden by parent:', parent);
                    break;
                }
                parent = parent.parentElement;
            }
            
            // Make it visible if hidden
            if (field.offsetParent === null) {
                console.log('  🔧 Attempting to make visible...');
                field.style.display = 'block';
                field.style.visibility = 'visible';
                field.style.opacity = '1';
                field.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    });
    
    // Check for MUI Cards
    const cards = document.querySelectorAll('.MuiCard-root');
    console.log(`\nTotal cards found: ${cards.length}`);
    cards.forEach((card, i) => {
        const title = card.querySelector('.MuiCardHeader-title')?.textContent || 'No title';
        console.log(`Card ${i}: ${title}`);
        
        // Check if card has text areas
        const textareas = card.querySelectorAll('textarea');
        if (textareas.length > 0) {
            console.log(`  Has ${textareas.length} textarea(s)`);
            textareas.forEach(ta => {
                const label = ta.parentElement?.querySelector('label')?.textContent || 'No label';
                console.log(`    - ${label} (${ta.value?.length || 0} chars)`);
            });
        }
    });
    
    // Check viewport and scroll position
    console.log('\n📏 Viewport info:');
    console.log('  Window height:', window.innerHeight);
    console.log('  Document height:', document.body.scrollHeight);
    console.log('  Current scroll:', window.scrollY);
    
    // Check for any overflow hidden that might be cutting off content
    const containers = document.querySelectorAll('[class*="Container"], [class*="Box"]');
    containers.forEach(container => {
        const overflow = window.getComputedStyle(container).overflow;
        if (overflow === 'hidden') {
            console.log('⚠️ Container with overflow hidden:', container);
        }
    });
}

// Run the debug function
debugIDCConfigDisplay();

// Also log React component info if available
if (window.React && window.React.version) {
    console.log('\nReact version:', window.React.version);
}

console.log('\n💡 To test if fields are working, run:');
console.log('document.querySelectorAll("textarea").forEach(ta => ta.style.border = "3px solid red")');