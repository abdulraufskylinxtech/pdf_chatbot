// CSRF Helper
function getCSRF() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}


// Tabs
function showTab(tabId) {
    document.querySelectorAll('.tab-section').forEach(t => t.style.display='none');
    document.getElementById(tabId).style.display='block';
}


// Close Modals
function closeModal() { document.getElementById('modal').style.display='none'; }
function closeFileModal() { document.getElementById('fileModal').style.display='none'; }
window.onclick = function(e){
    if(e.target.id==='modal') closeModal();
    if(e.target.id==='fileModal') closeFileModal();
}

// Users

function openCreateModal() {
    document.getElementById("createUserModal").style.display = "block";
}

function closeCreateModal() {
    document.getElementById("createUserModal").style.display = "none";
}

function openUpdateUserModal(id, username, email, is_main) {
    document.getElementById('updateUserModal').style.display = 'block';
    document.getElementById('updateUserId').value = id;
    document.getElementById('updateUsername').value = username;
    document.getElementById('updateEmail').value = email;
    document.getElementById('updateIsMain').checked = is_main;


    const form = document.getElementById('updateUserForm');
    form.action = `/admin/update-user/${id}/`;
}

function closeUpdateUserModal() {
    document.getElementById('updateUserModal').style.display = 'none';
}

// ----------------- CREATE -----------------
function openSaveCreateModal() {
    document.getElementById("saveCreateModal").style.display = "block";
}
function closeSaveCreateModal() {
    document.getElementById("saveCreateModal").style.display = "none";
}
function submitCreateForm() {
    document.getElementById("createUserForm").submit();
}

// ----------------- UPDATE -----------------
function openSaveUpdateModal() {
    document.getElementById("saveUpdateModal").style.display = "block";
}
function closeSaveUpdateModal() {
    document.getElementById("saveUpdateModal").style.display = "none";
}
function submitUpdateForm() {
    document.getElementById("updateUserForm").submit();
}

function submitForm() {
    if(currentFormId) {
        document.getElementById(currentFormId).submit();
        document.getElementById("createUserModal").style.display = "none";
        closeSaveModal(); 
    }
}


function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}


document.querySelectorAll('.is-main-checkbox').forEach(ch => {
    ch.addEventListener('change', function(){
        const userId = this.dataset.userId;
        const val = this.checked;
        fetch(`/admin/update-user/${userId}/`, {
            method: 'POST',
            headers: {'Content-Type':'application/json','X-CSRFToken':getCSRF()},
            body: JSON.stringify({is_main: val})
        }).then(r=>r.json()).then(d=>{
            if(d.success) {
                this.previousElementSibling.innerHTML = val ? '✓' : '✖';
            } else alert('Update failed');
        });
    });
});


function openDeleteModal(url) {
    document.getElementById("deleteModal").style.display = "block";
    document.getElementById("deleteForm").action = url;
}

function closeModal() {
    document.getElementById("deleteModal").style.display = "none";
}


function openSaveModal() {
    document.getElementById("saveModal").style.display = "flex";
}

function closeSaveModal() {
    document.getElementById("saveModal").style.display = "none";
}

function submitMainForm() {
    document.getElementById("mainForm").submit();
}


// Files


function openCreateFileModal() { document.getElementById("createFileModal").style.display = "block"; }
function closeCreateFileModal() { document.getElementById("createFileModal").style.display = "none"; }


function openUpdateFileModal(fileId, category) {
    document.getElementById('updateFileId').value = fileId;
    document.getElementById('updateFileCategory').value = category;
    document.getElementById('updateFileInput').value = '';
    document.getElementById('updateFileModal').style.display = 'block';
}
function closeUpdateFileModal() { document.getElementById("updateFileModal").style.display = "none"; }


function openSaveCreateFileModal() { document.getElementById("saveCreateFileModal").style.display = "flex"; }
function closeSaveCreateFileModal() { document.getElementById("saveCreateFileModal").style.display = "none"; }


function openSaveUpdateFileModal() {
    document.getElementById('saveUpdateFileModal').style.display = 'block';
}

function closeSaveUpdateFileModal() { document.getElementById("saveUpdateFileModal").style.display = "none"; }


function submitCreateFileForm() { document.getElementById("createFileForm").submit(); }
function submitUpdateFileForm() {
    document.getElementById('updateFileForm').submit();
}


function openDeleteFileModal(url) {
    const modal = document.getElementById('deleteFileModal');
    const form = document.getElementById('deleteFileForm');
    form.action = url;
    modal.style.display = 'block';
}

function closeDeleteFileModal() {
    document.getElementById('deleteFileModal').style.display = 'none';
}


// Chatbot Q&A


function openCreateChatModal() {
    document.getElementById("createChatModal").style.display = "flex";
}
function closeCreateChatModal() {
    document.getElementById("createChatModal").style.display = "none";
}
function openSaveCreateChatModal() {
    document.getElementById("saveCreateChatModal").style.display = "flex";
}
function closeSaveCreateChatModal() {
    document.getElementById("saveCreateChatModal").style.display = "none";
}
function submitCreateChatForm() {
    document.getElementById("createChatForm").submit();
}


function openUpdateChatModal(chatId, userId, fileId, question, answer) {
    document.getElementById("updateChatId").value = chatId;
    document.getElementById("updateChatUser").value = userId;
    document.getElementById("updateChatFile").value = fileId || "";
    document.getElementById("updateChatQuestion").value = question;
    document.getElementById("updateChatAnswer").value = answer;

    document.getElementById("updateChatForm").action = `/admin/update-chat/${chatId}/`;
    document.getElementById("updateChatModal").style.display = "flex";
}
function closeUpdateChatModal() {
    document.getElementById("updateChatModal").style.display = "none";
}
function openSaveUpdateChatModal() {
    document.getElementById("saveUpdateChatModal").style.display = "flex";
}
function closeSaveUpdateChatModal() {
    document.getElementById("saveUpdateChatModal").style.display = "none";
}
function submitUpdateChatForm() {
    document.getElementById("updateChatForm").submit();
}


function openDeleteChatModal(url) {
    if(confirm("Are you sure you want to delete this chat?")) {
        window.location.href = url;
    }
}



document.querySelectorAll('.update-user-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const id = this.dataset.userId;
        const username = this.dataset.username;
        const email = this.dataset.email;
        const isMain = this.dataset.isMain === "true";
        openUpdateUserModal(id, username, email, isMain);
    });
});

document.querySelectorAll('.update-file-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const fileId = this.dataset.fileId;
        const category = this.dataset.category;
        openUpdateFileModal(fileId, category);
    });
});


document.querySelectorAll('.update-chat-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const chatId = this.dataset.chatId;
        const userId = this.dataset.userId;
        const fileId = this.dataset.fileId;
        const question = this.dataset.question;
        const answer = this.dataset.answer;

       
        document.getElementById("updateChatId").value = chatId;
        document.getElementById("updateChatUser").value = userId;
        document.getElementById("updateChatFile").value = fileId;
        document.getElementById("updateChatQuestion").value = question;
        document.getElementById("updateChatAnswer").value = answer;

        
        document.getElementById("updateChatForm").action = `/admin/update-chat/${chatId}/`;

   
        document.getElementById("updateChatModal").style.display = "flex";
    });
});


function closeUpdateChatModal() {
    document.getElementById("updateChatModal").style.display = "none";
}

//     // Filter chunks by file
// function filterChunksByFile() {
//     const select = document.getElementById("filterFileSelect");
//     const selectedFileId = select.value;
//     const rows = document.querySelectorAll("#chunksTbody tr");
//     rows.forEach(row => {
//         const rowFileId = row.getAttribute("data-file-id");
//         row.style.display = (!selectedFileId || selectedFileId === rowFileId) ? "" : "none";
//     });
// }

// // Edit modal functions
// function openEditChunkModal(chunkId, chunkText) {
//     document.getElementById("editChunkId").value = chunkId;
//     document.getElementById("editChunkText").value = chunkText;
//     document.getElementById("editChunkModal").style.display = "block";
// }
// function closeEditChunkModal() {
//     document.getElementById("editChunkModal").style.display = "none";
// }

// // Delete modal functions
// function openDeleteChunkModal(url) {
//     const form = document.getElementById("deleteChunkForm");
//     form.action = url;
//     document.getElementById("deleteChunkModal").style.display = "block";
// }
// function closeDeleteChunkModal() {
//     document.getElementById("deleteChunkModal").style.display = "none";
// }


function updateUser() {
    const id = document.getElementById("userIdInput").value;
    const username = document.getElementById("usernameInput").value;
    const email = document.getElementById("emailInput").value;
    const isMain = document.getElementById("isMainCheckbox").checked;

    fetch(`/admin/update-user/${id}/`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken() 
        },
        body: JSON.stringify({
            username: username,
            email: email,
            is_main: isMain
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log(data);
       
    })
    .catch(err => console.error(err));
}


document.querySelectorAll('.is-main-checkbox').forEach(ch => {
    ch.addEventListener('change', function() {
        const userId = this.dataset.userId;
        const isChecked = this.checked;

        fetch(`/admin/update-user/${userId}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({ is_main: isChecked })
        })
        .then(res => res.json())
        .then(data => {
            if(data.success) {
                
                const symbolSpan = this.previousElementSibling;
                symbolSpan.innerHTML = isChecked ? '✓' : '✖';
            } else {
                alert('Update failed!');
                this.checked = !isChecked; 
            }
        })
        .catch(err => {
            console.error(err);
            this.checked = !isChecked; 
        });
    });
});
