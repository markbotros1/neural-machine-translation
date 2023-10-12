$("#modern").keyup(function() {
    
    var characterCount = $(this).val().length,
        current = $("#current");
        
    current.text(characterCount);

    if (characterCount > 25) {
        current.css("color", "#fb1010");
    } else {
        current.css("color","#666");
    }  
});
