
call plug#begin('~/.vim/plugged')

Plug 'klen/python-mode'

call plug#end()


set noerrorbells		" Disable bell
set visualbell
set t_vb=
set autochdir			" Current file determines $pwd
set number				" Show line numbers
set autoindent
set hlsearch			" Highlight results while typing
set incsearch			" Goto, and highlight matches live
set tabstop=4			" Tabs are displayed as # spaces
set shiftwidth=4		" Vim interprets tabs as # spaces
set wildmenu			" List all options when autocompleting
set showcmd				" Show details on current command
set scrolloff=5			" Always have # lines above/below cursor
set laststatus=2		" Always show status-line
set whichwrap=b,s		" Only allow backspace and space to traverse line endings
set ignorecase			" Non case-sensitive search,
set smartcase			" Except when using capital letters
set virtualedit=block	" Allow extending beyond content in block mode
set foldlevelstart=99	" When editing files with folds, start unfolded
set pastetoggle=<F9>	" Toggles paste mode
set synmaxcol=1000		" Lines longer than # will not be syntax highlighted
set splitright			" When splitting a window, new window goes on the right

" Enable mouse
set mouse=a

" When using diff, start in vertical split mode, and fill lines to
" sync lines
set diffopt=filler,vertical

" Allow backspace to work on newlines etc.
set backspace=indent,eol,start

" When using 'set list', use the following representation for whitespace
set listchars=eol:$,tab:>-,trail:~,extends:>,precedes:<

syntax on
